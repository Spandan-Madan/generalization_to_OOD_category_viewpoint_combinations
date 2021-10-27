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

project_root = os.getcwd()
data_root = "%s/data"%project_root

def make_dataset(list_file):
        images = []
        labels = []

        with open(list_file,'r') as F:
            lines = F.readlines()

        for line in lines:
            image = line.rstrip()
            images.append("%s/%s"%(data_root,image))
            label = image.replace('images/frame','labels/label_frame')
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

        impath = self.samples[index]
        #if 'om2' in impath:
        #    impath = impath.replace('/om2/user/smadan/car_dataset/','/data/graphics/toyota-pytorch/biased_dataset_generalization/datasets/')
        #if 'om5' in impath:
        #    impath = impath.replace('/om5/user/smadan/car_dataset/','/data/graphics/toyota-pytorch/biased_dataset_generalization/datasets/')
        sample_label = self.attributes[impath.split('/')[-1]]
        label_path = impath.replace('images/frame','labels/label_frame')

        #impath = impath.replace('/data/graphics/toyota-pytorch/biased_dataset_generalization/datasets/','/om5/user/smadan/car_dataset/')
        sample = Image.open(impath)
        
        
        
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
