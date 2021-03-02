import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys
import numpy as np
import torch
from skimage import feature


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

def format_label(imarray):
    imarray = imarray[0,:,:]
    # imarray[imarray==255]=20
    imarray[imarray<130] = 20
    # imarray[imarray>150] = 255
    # imarray[imarray==255] = 1

    class_count = len(np.unique(np.array(list(category_to_class_number.values()))))
    for val in labelval_to_category.keys():
        if labelval_to_category[val] in category_to_class_number.keys():
            new_val = category_to_class_number[labelval_to_category[val]]
            imarray[imarray == val] = new_val
#             print(new_val)
#             print('pixsum:',torch.sum(imarray == new_val).item())
#             print('pixfrac:',torch.sum(imarray == new_val).item()*100/float(224*224))
            if torch.sum(imarray == new_val).item()*100/float(224*224) < 0.3:
#                 print('Replacing')
#                 print('pixfrac:',torch.sum(imarray == new_val)*100/float(224*224))
                imarray[imarray == new_val] = 20
#             print('20 val:',torch.sum(imarray == 20).item()*100/float(224*224))
    imarray[imarray>class_count - 1] = 20
    label_count = np.zeros(20)
    for i in range(20):
        ct = torch.sum(imarray==i)
        label_count[i] = ct
    # print(label_count)

    car_label = np.argsort(label_count)[-1]
    return imarray, car_label


# def get_car_number(imarray):
#     imarray = imarray[0,:,:]
#     imarray[imarray<130] = 20

#     class_count = len(np.unique(np.array(list(category_to_class_number.values()))))
#     for val in labelval_to_category.keys():
#         if labelval_to_category[val] in category_to_class_number.keys():
#             new_val = category_to_class_number[labelval_to_category[val]]
#             imarray[imarray == val] = new_val
#             if torch.sum(imarray == new_val).item()*100/float(224*224) < 0.3:
#                 imarray[imarray == new_val] = 20

#     imarray[imarray>class_count - 1] = 20

def noisy_image(image_array,label_mask,car_label):
    image_array_copy = image_array.clone()

    car_mask = label_mask == car_label.item()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    edges_normed = np.zeros((3,224,224))
    for i in range(3):
        random_noise = np.clip(5 * np.random.randn(224,224) + 50,0,255)
        edges_normed[i,:,:] = (random_noise - mean[i].item())/std[i].item()
    xs,ys = np.where(car_mask==1)
    for i in range(len(xs)):
        x,y = xs[i],ys[i]
        image_array_copy[0,x,y] = edges_normed[0,x,y]/255.0
        image_array_copy[1,x,y] = edges_normed[1,x,y]/255.0
        image_array_copy[2,x,y] = edges_normed[2,x,y]/255.0
    return image_array_copy



# def noisy_image(image_array,label_mask,car_label):
# #     print(label_mask)
# #     print(car_label)
# #     print(image_array.shape)
#     image_array_copy = image_array.clone()
#
#     car_mask = label_mask == car_label.item()
#     # random_noise = []
#     # random_noise_0 = np.clip(5 * np.random.randn(224,224) + 100,0,255)
#     # random_noise_1 = np.clip(5 * np.random.randn(224,224) + 100,0,255)
#     # random_noise_2 = np.clip(5 * np.random.randn(224,224) + 100,0,255)
#     # random_noise = np.zeros((224,224,3))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#
#     random_noise_normed = np.zeros((3,224,224))
#     for i in range(3):
#         random_noise = np.clip(5 * np.random.randn(224,224) + 100,0,255)
#         random_noise_normed[i,:,:] = (random_noise - mean[i].item())/std[i].item()
# # edges_normed = (edges_3 - mean)/std
#     xs,ys = np.where(car_mask==1)
#     for i in range(len(xs)):
#         x,y = xs[i],ys[i]
#         image_array_copy[0,x,y] = random_noise[x,y,0]/255.0
#         image_array_copy[1,x,y] = random_noise[x,y,1]/255.0
#         image_array_copy[2,x,y] = random_noise[x,y,2]/255.0
#     return image_array_copy

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
        # print(path)
        sample = self.loader(path)
        sample_label = self.loader(label_path)

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(sample_label)

        formatted_label,car_id = format_label(target*255)
        sample_1 = sample.clone()
        sample_2 = sample.clone()
        # noise_added_image = sample_1
        noise_added_image = noisy_image(sample_1,formatted_label,car_id)
#         edges_added_image,m, edg = edges_image(sample_2,formatted_label,car_id)
        # return sample, target*255
        return noise_added_image, car_id

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
