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

labelval_to_category = {60: 'building',
 1:'DUMMY',
 59: 'curb',
 11: 'humans',
 58: 'road',
 57: 'sidewalk',
 6: 'sky',
 16: 'trees',
 31: 'unknown',
 56: 'vegetation',
 255: 'Fordtest1957_clean',
 254: 'Evo_clean',
 253: 'Camaro_clean',
 252: 'gto67_clean',
 251: 'subaru_impreza_wrx_sti_08_clean',
 250: 'fiat500blandswap_clean',
 249: 'Shelby_clean',
 248: 'GolfMK4-Cycles-Jay-Hardy-2011_clean',
 247: 'Audi_R8_clean',
 246: 'HUMMER_clean',
 245: 'Honda_Civic_Type_R_clean',
 244: 'Volvo_clean',
 243: 'audia6_clean',
 242: 'mini_clean',
 241: 'porsche_911_clean',
 240: 'CVPI2005_clean',
 239: 'Porsche993_GT2_clean',
 238: 'suzuki_sx4_clean',
 237: 'rapide_clean',
 235: 'cooper_mini_clean'}

category_to_class_number = {'Fordtest1957_clean': 0,
 'Evo_clean': 1,
 'Camaro_clean': 2,
 'gto67_clean': 3,
 'subaru_impreza_wrx_sti_08_clean': 4,
 'fiat500blandswap_clean': 5,
 'Shelby_clean': 6,
 'GolfMK4-Cycles-Jay-Hardy-2011_clean': 7,
 'Audi_R8_clean': 8,
 'HUMMER_clean': 9,
 'Honda_Civic_Type_R_clean': 10,
 'Volvo_clean': 11,
 'audia6_clean': 12,
 'mini_clean': 13,
 'porsche_911_clean': 14,
 'CVPI2005_clean': 15,
 'Porsche993_GT2_clean': 16,
 'suzuki_sx4_clean': 17,
 'rapide_clean': 18,
 'cooper_mini_clean': 19,
 'bg':20}

class_num_to_category = {}
category_to_pixel_val = {}
for cat in category_to_class_number.keys():
    val = category_to_class_number[cat]
    class_num_to_category[val] = cat

for pix_val in labelval_to_category.keys():
    val = labelval_to_category[pix_val]
    category_to_pixel_val[val] = pix_val

def make_dataset(list_file):
        images = []
        labels = []

        with open(list_file,'r') as F:
            lines = F.readlines()

        for line in lines:
            image = line.rstrip()
            images.append(image)
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
        label_path = impath.replace('images/frame','labels/label_frame')
        label_image = Image.open(label_path)


        sample_with_bg = Image.open(impath)
        sample_label = self.attributes[impath]

        car_label = sample_label[3]
        car_pixel_value = category_to_pixel_val[class_num_to_category[car_label]]
        imarr = np.array(sample_with_bg)
        label_arr = np.array(label_image)[:,:,0]
        mask = label_arr == car_pixel_value
        imarr[~mask] = 0
        
        sample = Image.fromarray(imarr)

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
