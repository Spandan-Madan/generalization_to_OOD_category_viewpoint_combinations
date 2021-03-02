import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import numpy as np
import torch
import cv2
import pickle

print(__file__)
loader_folder = '/'.join(__file__.split('/')[:-1])


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

# (-0.001, 391.2], (391.2, 1062.8], (1062.8, 1617.0], (1617.0, 2260.8], (2260.8, 3233.5], (3233.5, 5451.6], (5451.6, 9454.2], (9454.2, 17379.0], (17379.0, 21545.1], (21545.1, 31136.0]

#BIN_CUTOFFS = [0, 391, 1062, 1617, 2260, 3233, 5451, 9454, 17379, 21545, 31136] #### STARTS WITH 0 ALWAYS.

BIN_CUTOFFS = [0, 1062, 2260, 5451, 17379, 31136] ####STARTS WITH 0 ALWAYS.

def get_area_bin(bin_cutoffs,area):
    for i in range(len(bin_cutoffs)):
        val = bin_cutoffs[i]
        if area < val:
            break
    bin_value = i-1
    return bin_value

# with open('/data/graphics/toyota-pytorch/training_scaffold_own/res/loader/run_name_to_color_dict.p','rb') as F:
#     run_name_to_color_dict = pickle.load(F)
# /data/graphics/toyota-pytorch/training_scaffold_own/res/loader/
print(loader_folder)
with open('%s/recovered_angles_name_corrected.p'%loader_folder,'rb') as F:
    recovered_angles = pickle.load(F)
    
def format_label(imarray):
    imarray = imarray[0,:,:]
    imarray[imarray<130] = 20

    class_count = len(np.unique(np.array(list(category_to_class_number.values()))))
    for val in labelval_to_category.keys():
        if labelval_to_category[val] in category_to_class_number.keys():
            new_val = category_to_class_number[labelval_to_category[val]]
            imarray[imarray == val] = new_val
            if torch.sum(imarray == new_val).item()*100/float(224*224) < 0.3:
                imarray[imarray == new_val] = 20

    imarray[imarray>class_count - 1] = 20

    label_count = np.zeros(20)
    for i in range(20):
        ct = torch.sum(imarray==i)
        label_count[i] = ct
    # print(label_count)
    car_label = np.argsort(label_count)[-1]
    car_mask = np.array(imarray==int(car_label))
    contours, hierarchy = cv2.findContours(car_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx =0 
    largest_area = 0
    largest_centre = (-1,-1)
    for cnt in contours:
        idx += 1
        x,y,w,h = cv2.boundingRect(cnt)
        roi = car_mask[y:y+h,x:x+w]
        centre = (y + int(h/2), x + int(w/2))
        area = h*w
        if area > largest_area:
            largest_area = area
            largest_centre = centre
    area_bin = get_area_bin(BIN_CUTOFFS, largest_area)
    
        
# #     print(car_label)
#     try:
#         imarray.data == car_label
#         print('all good')
#     except:
#         imarray == car_label
#         print('car_label is ',car_label)
#         print(imarray.type())
#         print(imarray.shape())
    
#     try:
#         car_mask = imarray == car_label
#     except:
#         print('shite')
#         car_mask = torch.zeros((224,224))
    return car_label, area_bin, largest_centre,car_mask

# def format_label(imarray):
#     imarray = imarray[0,:,:]
#     new_label = np.zeros((imarray.shape[0],imarray.shape[1]))
#
#     for val in labelval_to_category.keys():
#         imarray[imarray==val] = category_to_class_number[labelval_to_category[val]]
#
#     # class_count = len(np.unique(np.array(list(category_to_class_number.values()))))
#     # print('Found a total %s'%class_count)
#     # imarray[imarray>class_count-1] = 0
#     # label_size = imarray.shape[0]
#     # num_classes = class_count
#     # formatted_label = np.zeros((class_count, label_size, label_size))
#     # print(formatted_label.shape)
#     # for i in range(class_count):
#     #     formatted_label[i] = new_label==i
#
# #     return formatted_label
#     return imarray


# def format_label_multi_category(imarray):
#     imarray = imarray[0,:,:]
#     label_size = imarray.shape[0]
#     num_classes = len(np.unique(imarray))
#     formatted_label = np.zeros((num_classes, label_size, label_size))
#     for i in range(num_classes):
#         formatted_label[i] = imarray==i
#
#     return formatted_label
# This below code was for old dataset labels. In that cars were labelled 1-136. In new rendering, cars are labelled 250,249,248 ... so on for every car added. Other objects are added as 1, 14, 27... (+13) for every object. 105 is the max label for objects other than cars.
#     imarray = imarray[0,:,:]
#     imarray[imarray==0] = 255
#     imarray[imarray<137] = 1
#     imarray[imarray>1] = 0
#     return imarray


# Uncommented because the new dataset format has changed and will now be constant. so, don't need this flexible function.
# def image_path_to_label_path(impath,LABEL_FOLDER):
#     image_name = impath.split('/')[-1]
#     object_name = impath.split('/')[-2]
#     phase_name = impath.split('/')[-3]
#     label_path = '%s/%s'%(LABEL_FOLDER,"label_"+image_name)
#     print(label_path)
#     return label_path


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return "label" not in filename_lower


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, extensions):
    images = []
    dir = os.path.expanduser(dir)

    image_folder = dir + 'images/'
    label_folder = dir + 'labels/'

    for image_file in sorted(os.listdir(image_folder)):
        label_file = 'label_' + image_file

        image_file_path = image_folder + image_file
        label_file_path = label_folder + label_file

        if '.png' in image_file_path and '.png' in label_file_path:
            item = (image_file_path,label_file_path)
            images.append(item)
    return images

class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
#         classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        #self.classes = classes
#         self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, label_path = self.samples[index]
        im_name = path.split('/')[-1]
        angles = recovered_angles[im_name]
        angles_bin = [min(9,max(0,int(angles[0]/10))),min(4,max(0,int(angles[1]/10)))]
        
        exp_name = '_'.join(path.split('.')[0].split('_')[-2:])
        car_color = 0
        
#         car_color_name = path.split('/')[-1].split('_')[1]
#         color_name_to_id = {'RED':0,'GREEN':1,'BLUE':2,'BLACK':3}
#         car_color = color_name_to_id[car_color_name]
        
        
        
        # print(path)
        sample = self.loader(path)
        sample_label = self.loader(label_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(sample_label)

        formatted_label,formatted_bin,formatted_centre,formatted_mask = format_label(target*255)
        # return sample, target*255
        return sample, formatted_label, formatted_bin, formatted_centre , car_color, path, angles_bin[0], angles_bin[1]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


# class DatasetFolder_with_paths(data.Dataset):
#     """A generic data loader where the samples are arranged in this way: ::
#
#         root/class_x/xxx.ext
#         root/class_x/xxy.ext
#         root/class_x/xxz.ext
#
#         root/class_y/123.ext
#         root/class_y/nsdf3.ext
#         root/class_y/asd932_.ext
#
#     Args:
#         root (string): Root directory path.
#         loader (callable): A function to load a sample given its path.
#         extensions (list[string]): A list of allowed extensions.
#         transform (callable, optional): A function/transform that takes in
#             a sample and returns a transformed version.
#             E.g, ``transforms.RandomCrop`` for images.
#         target_transform (callable, optional): A function/transform that takes
#             in the target and transforms it.
#
#      Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         samples (list): List of (sample path, class_index) tuples
#         targets (list): The class_index value for each image in the dataset
#     """
#
#     def __init__(self, root, loader, extensions, transform=None, target_transform=None):
# #         classes, class_to_idx = self._find_classes(root)
#         samples = make_dataset(root, extensions)
#         if len(samples) == 0:
#             raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
#                                "Supported extensions are: " + ",".join(extensions)))
#
#         self.root = root
#         self.loader = loader
#         self.extensions = extensions
#
#         #self.classes = classes
# #         self.class_to_idx = class_to_idx
#         self.samples = samples
#         self.targets = [s[1] for s in samples]
#
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def _find_classes(self, dir):
#         """
#         Finds the class folders in a dataset.
#
#         Args:
#             dir (string): Root directory path.
#
#         Returns:
#             tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#
#         Ensures:
#             No class is a subdirectory of another.
#         """
#         if sys.version_info >= (3, 5):
#             # Faster and available in Python 3.5 and above
#             classes = [d.name for d in os.scandir(dir) if d.is_dir()]
#         else:
#             classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
#         classes.sort()
#         class_to_idx = {classes[i]: i for i in range(len(classes))}
#         return classes, class_to_idx
#
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index
#
#         Returns:
#             tuple: (sample, target) where target is class_index of the target class.
#         """
#         path, label_path = self.samples[index]
# #         print(path)
#         sample = self.loader(path)
#         sample_label = self.loader(label_path)
#
#
# #         reformatted_label = assign_pixel_val(label)
#
#         if self.transform is not None:
#             sample = self.transform(sample)
#         if self.target_transform is not None:
#             target = self.target_transform(sample_label)
#
#         formatted_label = format_label(target*255)
# #             single_channel = target[0,:,:]
#         return sample, formatted_label,path,label_path
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __repr__(self):
#         fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
#         fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
#         fmt_str += '    Root Location: {}\n'.format(self.root)
#         tmp = '    Transforms (if any): '
#         fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         tmp = '    Target Transforms (if any): '
#         fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
#         return fmt_str



IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class multi_category_ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(multi_category_ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


# class ImageFolder_with_paths(DatasetFolder_with_paths):
#     """A generic data loader where the images are arranged in this way: ::
#
#         root/dog/xxx.png
#         root/dog/xxy.png
#         root/dog/xxz.png
#
#         root/cat/123.png
#         root/cat/nsdf3.png
#         root/cat/asd932_.png
#
#     Args:
#         root (string): Root directory path.
#         transform (callable, optional): A function/transform that  takes in an PIL image
#             and returns a transformed version. E.g, ``transforms.RandomCrop``
#         target_transform (callable, optional): A function/transform that takes in the
#             target and transforms it.
#         loader (callable, optional): A function to load an image given its path.
#
#      Attributes:
#         classes (list): List of the class names.
#         class_to_idx (dict): Dict with items (class_name, class_index).
#         imgs (list): List of (image path, class_index) tuples
#     """
#     def __init__(self, root, transform=None, target_transform=None,
#                  loader=default_loader):
#         super(ImageFolder_with_paths, self).__init__(root, loader, IMG_EXTENSIONS,
#                                           transform=transform,
#                                           target_transform=target_transform)
#         self.imgs = self.samples
