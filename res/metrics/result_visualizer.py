from __future__ import print_function, division
import pickle
import torch
import sys
sys.path.append('../../res/')
from loader.loader import get_loader
from model.model import get_model
from optimizer.optimizer import get_optimizer
from loss.loss import get_loss
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
import sys
from PIL import Image
import torch.nn.functional as F
import random
import imageio
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from config_file import *

def render_comparison_gif(input_image,gt,pred,save_path):
    pred_label_numpy = pred.cpu().detach().numpy()
    gt_and_pred_images = []
    for i in range(7):
        input_image_tensor = input_image.cpu().detach()
        input_im = convert_input(input_image_tensor)*255
        input_im = np.concatenate((np.ones((50,224,3))*255,input_im),axis=0)
        
        gt_label_numpy = gt.cpu().detach().numpy()
        gt_label = (gt_label_numpy == i).astype('uint8') * 255
        gt_label = np.concatenate(((np.ones((50,224))*255),gt_label))


        pred_label = pred_label_numpy[i]
        pred_label[pred_label>0.7] = 255
        pred_label[pred_label<0.7] = 0
        pred_label = np.concatenate(((np.ones((50,224))*255),pred_label),axis=0)
        pred_label = pred_label.astype('uint8')
        
        
        
        gt_label_repeated = np.repeat(gt_label[:, :, np.newaxis], 3, axis=2)
        pred_label_repeated = np.repeat(pred_label[:, :, np.newaxis], 3, axis=2)
        
        
    
        gt_and_pred = np.concatenate((input_im, gt_label_repeated, pred_label_repeated),axis=1)
        gt_and_pred_text_bar = np.concatenate((gt_and_pred,np.ones((50,672,3))*255),axis=0)
        gt_and_pred_image = Image.fromarray(gt_and_pred_text_bar.astype('uint8'))

        im = gt_and_pred_image

        caption = class_number_to_category[i]
        caption_font = ImageFont.truetype("/afs/csail.mit.edu/u/s/smadan/arial.ttf", 24)
        column_caption_0 = 'Image'
        column_caption_1 = 'Ground Truth'
        column_caption_2 = 'Model Prediction'
        column_font = ImageFont.truetype("/afs/csail.mit.edu/u/s/smadan/arial.ttf", 16)

        color = 'black'
        draw = ImageDraw.Draw(im)

        caption_w, caption_h = draw.textsize(caption,font=caption_font)
        caption_position = (336-caption_w/2,280)
        draw.text(caption_position, caption, font=caption_font,fill=color)
        
        column_0_w,column_0_h = draw.textsize(column_caption_0,font=column_font)
        column_0_position = (112-column_0_w/2,20)
        draw.text(column_0_position, column_caption_0, font=column_font,fill=color)
        
        column_1_w,column_1_h = draw.textsize(column_caption_1,font=column_font)
        column_1_position = (336-column_1_w/2,20)
        draw.text(column_1_position, column_caption_1, font=column_font,fill=color)

        column_2_w,column_2_h = draw.textsize(column_caption_2,font=column_font)
        column_2_position = (560-column_2_w/2,20)
        draw.text(column_2_position, column_caption_2, font=column_font,fill=color)

        gt_and_pred_images.append(im)

    imageio.mimsave(save_path,gt_and_pred_images,duration=2)
    
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IN_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(IN_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
   ])
}

label_transforms = {
    'train': transforms.Compose([
        transforms.Resize(IN_SIZE),
        transforms.ToTensor()
    ]),
    'val': transforms.Compose([
        transforms.Resize(IN_SIZE),
        transforms.ToTensor()
   ])
}

loader = get_loader(LOADER_TYPE)


TRAIN_DIR = '/data/graphics/toyota-pytorch/biased_dataset_generalization/datasets/' + DATASET_NAME + '/train/'
VAL_DIR = '/data/graphics/toyota-pytorch/biased_dataset_generalization/datasets/' + DATASET_NAME + '/val/'


dset_train = loader(TRAIN_DIR, data_transforms['train'],target_transform=label_transforms['train'])
dset_val = loader(VAL_DIR, data_transforms['val'],target_transform=label_transforms['val'])


train_loader = torch.utils.data.DataLoader(dset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,drop_last=True)
val_loader = torch.utils.data.DataLoader(dset_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,drop_last=True)

dset_loaders = {'train':train_loader,'val':val_loader}
dset_sizes = {}
dset_sizes['train'] = len(dset_train)
dset_sizes['val'] = len(dset_val)


trained_model = torch.load('saved_models/%s.pt'%EXPERIMENT_ID)

category_to_class_number = {'building': 2,
 'curb': 3,
 'humans': 4,
 'road': 5,
 'sidewalk': 3,
 'sky': 0,
 'trees': 6,
 'unknown': 0,
 'background/sky/unknown':0,
 'vegetation': 6,
 'car':1}

class_number_to_category = {}
for key in category_to_class_number.keys():
    val = category_to_class_number[key]
    class_number_to_category[val] = key

GIF_SAVE_FOLDER = 'prediction_gifs/'

counter = 0
for data in dset_loaders['val']:
    inputs, labels = data
    resized_in = inputs[:,:,:,:224]
    resized_labels = labels[:,:,:224]
    input_var = Variable(resized_in.float().cuda())                             
    label_var = Variable(resized_labels.long().cuda())               
    outputs = trained_model(input_var)  
    out_cpu = outputs.cpu().detach()
    for i in range(resized_in.shape[0]):
        save_path = 'prediction_%s.gif'%counter
        render_comparison_gif(resized_in[i],resized_labels[i],out_cpu[i],save_path)
        counter += 1

