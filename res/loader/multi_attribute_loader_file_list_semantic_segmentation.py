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

labelval_to_category={60:'building',
                      59:'curb',
                      11:'humans',
                      58:'road',
                      57:'sidewalk',
                      0:'sky',
                      129:'trees',
                      31:'unknown',
                      56:'vegetation',
                      1:'unknown',
                      255:'car',
}

category_to_class_number = {
    'sky': 0,
    'building': 1,
    'humans': 2,
    'road': 3,
    'curb': 4,
    'sidewalk': 5,
    'trees': 6,
    'vegetation': 7,
    'unknown': 8,
    'car':9
}

def format_label(imarray):
    imarray = imarray[:,:,0]
    for val in labelval_to_category.keys():
        imarray[imarray==val] = category_to_class_number[labelval_to_category[val]]
        
    imarray[imarray>150] = 9
    
    label_size = imarray.shape[0]
    num_classes = 10
    formatted_label = np.zeros((num_classes, label_size, label_size))
    for i in range(num_classes):
        formatted_label[i] = imarray==i
    return formatted_label
    
def make_dataset(list_file, data_dir):
        images = []
        labels = []

        with open(list_file,'r') as F:
            lines = F.readlines()

        for line in lines:
            image = line.rstrip()
            images.append("%s/%s"%(data_dir,image))
            label = image.replace('images/frame','labels/label_frame')
            labels.append(label)


        return images, labels

class FileListFolder(data.Dataset):
    def __init__(self, file_list, attributes_dict, transform, data_dir):
        samples,targets  = make_dataset(file_list, data_dir)

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
        label_path = impath.replace('images/frame','labels/label_frame')
#         return impath, label_path
        sample = Image.open(impath)
        label = np.array(Image.open(label_path))

        if self.transform is not None:
            transformed_sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(label)
        
        formatted_label = format_label(label)
        return transformed_sample, formatted_label, impath

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
