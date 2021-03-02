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
        
        impath = self.samples[index]
        file_name = impath.split('/')[-1]
        category_id = file_name.split('_')[0]
        category_num = self.attributes[category_id]
        
        with open(impath,'rb') as F:
            sample = pickle.load(F)
#         sample = Image.open(impath)
#         sample.resize((224,224))
        print(sample.shape)
        sample_label = category_num
        print(self.transform)
        if self.transform is not None:
            transformed_sample = self.transform(sample)

#         transformed_label = torch.LongTensor(sample_label)

        return transformed_sample, sample_label, impath

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
