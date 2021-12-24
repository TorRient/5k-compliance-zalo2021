from tqdm.auto import tqdm
import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, auc, f1_score
from torch.cuda.amp import autocast

def acc_score(output, target):
    y_pred = torch.argmax(torch.exp(output), dim=1).cpu().detach().numpy()
    target = target.cpu()
    return accuracy_score(target, y_pred)

def train_fn(train_loader, model, model_clip, loss_fn, optimizer, epoch, device, scaler, scheduler=None):
    model.train()
    stream = tqdm(train_loader)
    final_targets = []
    final_outputs = []
    for i, (image, target) in enumerate(stream, start=1):
        with autocast():
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if model_clip:
                features_clip = model_clip.encode_image(image.to(device))
                output = model(image, features_clip)
            else:
                output = model(image)
            loss = loss_fn(output, target)
            scaler.scale(loss).backward()
            acc = acc_score(output, target)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            targets = (target.detach().cpu().numpy()).tolist()
            outputs = (torch.argmax(torch.exp(output), dim=1).detach().cpu().numpy()).tolist()
            
            final_targets.extend(targets)
            final_outputs.extend(outputs)
            stream.set_description(f"Epoch {epoch:02}. Train. ACC {acc} Loss {loss}")
    return final_targets, final_outputs

def validation_fn(validation_loader, model, model_clip, loss_fn, epoch, device):
    model.eval()
    stream = tqdm(validation_loader)
    final_targets = []
    final_outputs = []
    with torch.no_grad():
        for i, (image, target) in enumerate(stream, start=1):
            with autocast():
                image = image.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if model_clip:
                    features_clip = model_clip.encode_image(image.to(device))
                    output = model(image, features_clip)
                else:
                    output = model(image)
                loss = loss_fn(output, target)
                acc = acc_score(output, target)
                stream.set_description(f"Epoch: {epoch:02}. Valid. ACC {acc} Loss {loss}")
                    
                targets = (target.detach().cpu().numpy()).tolist()
                outputs = (torch.argmax(torch.exp(output), dim=1).detach().cpu().numpy()).tolist()
                
                final_targets.extend(targets)
                final_outputs.extend(outputs)
        
    return final_targets, final_outputs

def test_fn(test_loader, model, model_clip, device):
    model.eval()
    stream = tqdm(test_loader)
    final_outputs = []
    
    with torch.no_grad():
        for i, image in enumerate(stream, start=1):
            with autocast():
                image = image.to(device, non_blocking=True)
                if model_clip:
                    features_clip = model_clip.encode_image(image.to(device))
                    output = model(image, features_clip)
                else:
                    output = model(image)
                outputs = (torch.argmax(torch.exp(output), dim=1).detach().cpu().numpy()).tolist()
                final_outputs.extend(outputs)
        
    return final_outputs