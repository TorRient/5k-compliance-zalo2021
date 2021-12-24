import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, auc, f1_score
from model import PetNetONLY
from trainer import *
from sklearn import model_selection
from dataset_zalo import ZaloDataset, ZaloTestset
from transforms import train_transform_object, valid_transform_object
from config import *
import glob
import gc
from utils import create_csv_mask
import random
from label_smoothing import LabelSmoothingCrossEntropy
gc.enable()
from torch.cuda.amp import autocast, GradScaler
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir_model_mask", default="./train_mask", help="dir model mask")
parser.add_argument("--dir_train", default="./train", help="dir train data")
args = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
import clip
if SIZE_IMG == 224:
    model_name = "swin_large_patch4_window7_224"
elif SIZE_IMG == 384:
    model_name = "swin_large_patch4_window12_384"
# else:
def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(42)

print(SIZE_IMG)
model_clip = None
if model_clip:
    model_clip.eval()
    
print(model_name)
best_models_of_each_fold = []

task_name = "mask"

def get_dataset(df, images, state='training'):
    ids = list(df['image_id'])
    image_paths = [os.path.join(images, str(idx) + '.jpg') for idx in ids]
    target = df[task_name].values
    df.drop(["image_id","fname","mask","distancing","5k","kfold"], inplace=True, axis=1)

    if state == 'training':
        transform = train_transform_object(SIZE_IMG)
    elif state == 'validation' or state == 'testing':
        transform = valid_transform_object(SIZE_IMG)
    else:
        transform = None

    return ZaloDataset(image_paths, target, transform)

def get_dataset_test(df, images, state='testing'):
    ids = list(df['image_id'])
    image_paths = [os.path.join(images, str(idx) + '.jpg') for idx in ids]
    df.drop(["image_id","fname"], inplace=True, axis=1)

    if state == 'training':
        transform = train_transform_object(SIZE_IMG)
    elif state == 'validation' or state == 'testing':
        transform = valid_transform_object(SIZE_IMG)
    else:
        transform = None

    return ZaloTestset(image_paths, transform)

scaler = GradScaler()
model_dir = args.dir_model_mask
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
path_mask_csv_init = model_dir + '/train_{}folds_mask_first.csv'.format(FOLDS)
path_mask_csv_update = model_dir + '/train_{}folds_mask.csv'.format(FOLDS)
path_train_init = args.dir_train + '/train_meta.csv'
path_train_update = model_dir + '/train_meta_update.csv'
for loopp in range(1,2):
    print("LOOP: ", loopp)
    best_models_of_each_fold = []
    for fold in range(0, FOLDS):
        if loopp == 1:
            if not os.path.exists(path_mask_csv_init):
                print("CREATE FOLDS")
                create_csv_mask(path_mask_csv_init, path_train_init, task_name="mask")
            data = pd.read_csv(path_mask_csv_init)
        else:
            data = pd.read_csv(path_mask_csv_update)
        train = data[data['kfold'] != fold]
        print("LEN train: ", len(train[train[task_name] == 1]), len(train[train[task_name] == 0]))
        val = data[data['kfold'] == fold]
        print("LEN val: ", len(val))
        images = args.dir_train + '/images'

        train_dataset = get_dataset(train, images)
        val_dataset = get_dataset(val, images, state='validation')

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*4, shuffle=False)

        model_params = {
            'model_name' : model_name,
            'out_features' : 2,
            'inp_channels' : 3,
            'pretrained' : True
        }
        model = PetNetONLY(**model_params)
        model = model.to(device)

        loss_fn = LabelSmoothingCrossEntropy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-6, amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=MIN_LR, last_epoch=-1)
        best_f1 = -np.inf
        best_epoch = -np.inf
        best_model_name = None
        for epoch in range(EPOCHS):
            print("LR: ", scheduler.get_last_lr())
            train_targets, train_predictions = train_fn(train_loader, model, model_clip, loss_fn, optimizer, epoch, device, scaler)
            train_acc = round(accuracy_score(train_targets, train_predictions), 3)
            train_f1 = round(f1_score(train_targets, train_predictions), 3)
            train_conf_matric = confusion_matrix(train_targets, train_predictions)
            valid_targets, predictions = validation_fn(val_loader, model, model_clip, loss_fn, epoch, device)
            val_acc = round(accuracy_score(valid_targets, predictions), 3)
            val_f1 = round(f1_score(valid_targets, predictions), 3)
            val_conf_matric = confusion_matrix(valid_targets, predictions)
            print(f"TRAIN: F1 {train_f1} Acc {train_acc}")
            print(f"VAL: F1 {val_f1} Acc {val_acc}")
            print("CONFUSION MATRIX: ", val_conf_matric)
            scheduler.step()
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                if best_model_name is not None:
                    os.remove(best_model_name)
                if model_clip:
                    torch.save(model.state_dict(), model_dir + f"/{task_name}_clip_{model_params['model_name']}_{fold}_fold_{epoch}_epoch_{val_f1}_f1_{val_acc}_acc.pth")
                    best_model_name = model_dir + f"/{task_name}_clip_{model_params['model_name']}_{fold}_fold_{epoch}_epoch_{val_f1}_f1_{val_acc}_acc.pth"
                else:
                    torch.save(model.state_dict(), model_dir + f"/{task_name}_{model_params['model_name']}_{fold}_fold_{epoch}_epoch_{val_f1}_f1_{val_acc}_acc.pth")
                    best_model_name = model_dir + f"/{task_name}_{model_params['model_name']}_{fold}_fold_{epoch}_epoch_{val_f1}_f1_{val_acc}_acc.pth"
                print(f'The Best saved model is: {best_model_name}')
            
        best_models_of_each_fold.append(best_model_name)
        print(''.join(['#']*50))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    
    # Infer
    final_predict = []
    for fold in range(0, FOLDS):
        if loopp == 1:
            test = pd.read_csv(path_train_init)
        else:
            test = pd.read_csv(path_train_update)
        test = test[test[task_name].isna()]
        images = args.dir_train + '/images'
        val_dataset = get_dataset_test(test, images, state='testing')
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*4, shuffle=False)
        model_params = {
            'model_name' : model_name,
            'out_features' : 2,
            'inp_channels' : 3,
            'pretrained' : False
        }
        model = PetNetONLY(**model_params)
        if True:
            list_pretrained = glob.glob(model_dir + "/*.pth") #best_models_of_each_fold
            model_load = sorted(list_pretrained)[fold]
            print("LOAD MODEL: ", model_load)
            model.load_state_dict(torch.load(model_load))
        model = model.to(device)
        predictions = test_fn(val_loader, model, model_clip, device=device)
        final_predict.append(predictions)
    final_predict = np.asarray(final_predict)
    final_predict = np.mean(final_predict, axis=0)
    print(final_predict.shape)
    final = []
    for x in final_predict:
        if x >= 0.8:
            final.append(1)
        elif x <= 0.2:
            final.append(0)
        else:
            final.append(2)
    if loopp == 1:
        sub_csv = pd.read_csv(path_train_init)
    else:
        sub_csv = pd.read_csv(path_train_update)
    count = 0
    for i in range(len(sub_csv)):
        if sub_csv.loc[i, "mask"] != 0 and sub_csv.loc[i, "mask"] != 1:
            if final[count] != 2:
                sub_csv.loc[i, 'mask'] = final[count]
            count += 1
        if sub_csv.loc[i, "distancing"] != 0 and sub_csv.loc[i, "distancing"] != 1:
            if sub_csv.loc[i, "5k"] == 1 and sub_csv.loc[i, "mask"] == 1:
                sub_csv.loc[i, 'distancing'] = 1
            if sub_csv.loc[i, "5k"] == 0 and sub_csv.loc[i, "mask"] == 1:
                sub_csv.loc[i, 'distancing'] = 0

    sub_csv.to_csv(path_train_update, index=False)
    # split
    len_df = create_csv_mask(path_mask_csv_update, path_train_update, task_name="mask")
    loopp += 1