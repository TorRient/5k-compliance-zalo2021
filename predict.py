import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import PetNetONLY
from trainer import *
from dataset_zalo import ZaloTestset
from transforms import train_transform_object, valid_transform_object
from config import *
import glob
import gc
gc.enable()
import warnings
import sklearn.exceptions
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dir_model_mask", default="./train_mask", help="dir model path")
parser.add_argument("--dir_model_5k", default="./train_5k", help="dir model 5k")
parser.add_argument("--dir_test", default="./private_test", help="dir test")
parser.add_argument("--output_submit", default="./submission.csv", help="output submit")
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
model_clip = None
if model_clip:
    model_clip.eval()
best_models_of_each_fold = []
rmse_tracker = []

def get_dataset(df, images, state='testing'):
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

# Task mask
model_dir = args.dir_model_mask
final_predict = []
task_name = "mask"
for fold in range(0, FOLDS):
    test = pd.read_csv(args.dir_test + '/private_test_meta.csv')
    images = args.dir_test + '/images'
    val_dataset = get_dataset(test, images, state='testing')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*4, shuffle=False)
    model_params = {
        'model_name' : model_name,
        'out_features' : 2,
        'inp_channels' : 3,
        'pretrained' : False
    }
    model = PetNetONLY(**model_params)
    if True:
        if model_clip:
            list_pretrained = glob.glob(model_dir + "/" + task_name + "_clip_" + model_params["model_name"] + "*")
        else:
            list_pretrained = glob.glob(model_dir + "/" + task_name + "_" + model_params["model_name"] + "*")
        model_load = sorted(list_pretrained)[fold]
        print("LOAD MODEL: ", model_load)
        model.load_state_dict(torch.load(model_load))
    model = model.to(device)
    predictions = test_fn(val_loader, model, model_clip, device=device)
    final_predict.append(predictions)
final_predict = np.asarray(final_predict)
final_predict = np.mean(final_predict, axis=0)
print(final_predict.shape)
final_mask = []
for x in final_predict:
    if x > 0.25:
        final_mask.append(1)
    else:
        final_mask.append(0)

# Task 5k
model_dir = args.dir_model_5k
final_predict = []
task_name = "5k"
for fold in range(0, FOLDS):
    test = pd.read_csv(args.dir_test + '/private_test_meta.csv')
    images = args.dir_test + '/images'
    val_dataset = get_dataset(test, images, state='testing')
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*4, shuffle=False)
    model_params = {
        'model_name' : model_name,
        'out_features' : 2,
        'inp_channels' : 3,
        'pretrained' : False
    }
    model = PetNetONLY(**model_params)
    if True:
        if model_clip:
            list_pretrained = glob.glob(model_dir + "/" + task_name + "_clip_" + model_params["model_name"] + "*")
        else:
            list_pretrained = glob.glob(model_dir + "/" + task_name + "_" + model_params["model_name"] + "*")
        model_load = sorted(list_pretrained)[fold]
        print("LOAD MODEL: ", model_load)
        model.load_state_dict(torch.load(model_load))
    model = model.to(device)
    predictions = test_fn(val_loader, model, model_clip, device=device)
    final_predict.append(predictions)
final_predict = np.asarray(final_predict)
final_predict = np.mean(final_predict, axis=0)
final = []
for i, x in enumerate(final_predict):
    if final_mask[i] == 0:
        final.append(0)
    else:
        if x > 0.5:
            final.append(1)
        else:
            final.append(0)
sub_csv = pd.read_csv(args.dir_test + '/private_test_meta.csv')
for i in range(len(final)):
    sub_csv.loc[i, '5K'] = final[i]
sub_csv.to_csv(args.output_submit, index=False)
