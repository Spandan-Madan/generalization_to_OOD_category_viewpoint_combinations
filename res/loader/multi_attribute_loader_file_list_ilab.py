import torch.utils.data as data
from torchvision import datasets, models, transforms
IN_SIZE = 224
import pickle
from PIL import Image
import matplotlib.pyplot as plt
import os
import os.path
import sys
import numpy as np
import torch

def get_image_attributes(imfile):
    parts = imfile.split('.')[0].split('-')
    category = parts[0]
    obj_id = parts[1]
    obj_instance = "%s_%s"%(category,obj_id)
    
    background = parts[2]
    elev = parts[3]
    azimuth = parts[4]
    light = parts[5]
    focus = parts[6]
    
    return category, obj_instance, background, elev, azimuth, light, focus

def make_dataset(list_file):
        images = []
        labels = []

        with open(list_file,'r') as F:
            lines = F.readlines()

        for line in lines:
            image = line.rstrip()
            images.append(image)
            label = image
            labels.append(label)


        return images, labels

class FileListFolder(data.Dataset):
    def __init__(self, file_list, attributes_dict, transform):
        samples,targets  = make_dataset(file_list)

        if len(samples) == 0:
            raise(RuntimeError("Found 0 samples"))

        self.root = file_list

        self.samples = samples
        self.targets = targets

        self.transform = transform

        with open(attributes_dict, 'rb') as F:
            attributes = pickle.load(F)

        self.attributes = attributes


    def __getitem__(self, index):

        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        
        #instance_list = ['car_i0066', 'car_i0070', 'car_i0072', 'car_i0063', 'car_i0079', 'car_i0082', ]
        cats_list = ['bus','car','plane','heli','tank','monster']
        azimuths = ['r01', 'r02', 'r03', 'r04', 'r06', 'r07']
        
        impath = self.samples[index]
        label_path = impath

        impath = impath.replace('om2','om5')
        sample = Image.open(impath)
        imname = impath.split('/')[-1]
        
        category, obj_instance, background, elev, azimuth, light, focus = get_image_attributes(imname)
        
        cat_num = cats_list.index(category)
#         instance_num = instance_list.index(obj_instance)
        azimuth_num = azimuths.index(azimuth)
        
        sample_label = [0, azimuth_num, 0, cat_num]
        
        floated_labels = []
        for s in sample_label:
            floated_labels.append(float(s))

        if self.transform is not None:
            transformed_sample = self.transform(sample)

        transformed_labels = torch.LongTensor(floated_labels)

        return transformed_sample, transformed_labels, impath

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'

        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        # tmp = '    Target Transforms (if any): '
        # fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
