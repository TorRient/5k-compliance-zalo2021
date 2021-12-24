import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, auc, f1_score
from model import PetNetONLY
from trainer import *
from dataset_zalo import ZaloDataset
from transforms import train_transform_object, valid_transform_object
from config import *
import glob
import gc
import random
from utils import create_csv_mask
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
parser.add_argument("--dir_model_distancing", default="./train_distancing", help="dir model distancing")
parser.add_argument("--dir_model_5k", default="./train_5k", help="dir model 5k")
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
rmse_tracker = []

task_name = "5k"

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

scaler = GradScaler()
model_dir = args.dir_model_5k
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
path_5k_csv = model_dir + "/train_{}folds_5k.csv".format(FOLDS)
for fold in range(0, FOLDS):
    if not os.path.exists(path_5k_csv):
        create_csv_mask(path_5k_csv, args.dir_model_distancing + '/train_meta_update.csv', task_name="5k")
    data = pd.read_csv(path_5k_csv)
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
    # optimizer = bnb.optim.Adam8bit(model.parameters(), lr=LR, betas=(0.9, 0.995))
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
    rmse_tracker.append(best_f1)
    print(''.join(['#']*50))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    