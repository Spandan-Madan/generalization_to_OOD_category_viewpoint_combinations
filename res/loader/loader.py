import pickle
import os
from loader.synthetic_loader import ImageFolder
from loader.old_dataset_loader import ImageFolder as ImageFolder_OLD_DATASET
from loader.synthetic_loader import ImageFolder_with_paths
from loader.multi_category_synthetic_loader import multi_category_ImageFolder
from loader.cityscapes_loader import ImageFolder as ImageFolder_cityscapes
from loader.multi_category_per_car_synthetic_loader import multi_category_ImageFolder as multi_category_per_car_ImageFolder
from loader.multi_category_per_car_synthetic_loader_only_cars import multi_category_ImageFolder as multi_category_per_car_only_cars_ImageFolder
from loader.classification_loader import multi_category_ImageFolder as multi_category_ImageFolder_classification
from loader.multi_attribute_loader import multi_category_ImageFolder as multi_category_ImageFolder_multi_attribute
from loader.classification_loader_noisy_texture import multi_category_ImageFolder as multi_category_ImageFolder_classification_noisy_texture
from loader.classification_loader_edges_texture import multi_category_ImageFolder as multi_category_ImageFolder_classification_edges_texture
from loader.multi_attribute_loader_file_list import FileListFolder as multi_attribute_loader_file_list
from loader.multi_attribute_loader_file_list_no_bg import FileListFolder as multi_attribute_loader_file_list_no_bg
from loader.multi_attribute_loader_file_list_ilab import FileListFolder as multi_attribute_loader_file_list_ilab
from loader.multi_attribute_loader_file_list_uiuc import FileListFolder as multi_attribute_loader_file_list_uiuc
from loader.multi_attribute_loader_file_list_mnist_rotation import FileListFolder as multi_attribute_loader_file_list_mnist_rotation
from loader.multi_attribute_loader_file_list_mnist_rotation_29 import FileListFolder as multi_attribute_loader_file_list_mnist_rotation_29
from loader.multi_attribute_loader_file_list_shapenet import FileListFolder as multi_attribute_loader_file_list_shapenet
from loader.multi_attribute_loader_file_list_shapenet_tensor import FileListFolder as multi_attribute_loader_file_list_shapenet_tensor

from loader.PennFudanLoader import PennFudanDataset
import sys

print(__file__)
loader_folder = '/'.join(__file__.split('/')[:-1])
print(loader_folder)
with open('%s/category_to_class_number_synthetic.p'%loader_folder,'rb') as F:
    category_to_class_number_synthetic = pickle.load(F)


with open('%s/category_to_class_number_cityscapes.p'%loader_folder,'rb') as F:
    category_to_class_number_cityscapes = pickle.load(F)

with open('%s/category_to_class_number_per_car_synthetic.p'%loader_folder,'rb') as F:
    category_to_class_number_per_car_synthetic = pickle.load(F)

with open('%s/category_to_class_number_per_car_synthetic_only_cars.p'%loader_folder,'rb') as F:
    category_to_class_number_per_car_synthetic_only_cars = pickle.load(F)

with open('%s/category_to_class_number_per_car_synthetic_only_cars_no_police.p'%loader_folder,'rb') as F:
    category_to_class_number_per_car_synthetic_only_cars_no_police = pickle.load(F)


with open('%s/category_to_class_number_ilab_5_cars.p'%loader_folder,'rb') as F:
    category_to_class_number_ilab_5_cars = pickle.load(F)

def get_loader(name):
    """get_loader
    :param name:
    """
    return {
        "synthetic_binary_segmentation": ImageFolder,
        "synthetic_multi_category_segmentation": multi_category_ImageFolder,
        "synthetic_multi_category_per_car_segmentation": multi_category_per_car_ImageFolder,
        "synthetic_multi_category_per_car_segmentation_only_cars": multi_category_per_car_only_cars_ImageFolder,
        "old_dataset_segmentation" : ImageFolder_OLD_DATASET,
        "segmentation_with_paths" : ImageFolder_with_paths,
        "cityscapes" : ImageFolder_cityscapes,
        "classification_per_car_model" : multi_category_ImageFolder_classification,
        "multi_attribute_loader" : multi_category_ImageFolder_multi_attribute,
        "classification_per_car_model_noisy_texture" : multi_category_ImageFolder_classification_noisy_texture,
        "classification_per_car_model_edges_texture" : multi_category_ImageFolder_classification_edges_texture,
        "multi_attribute_loader_file_list" : multi_attribute_loader_file_list,
        "multi_attribute_loader_file_list_no_bg" : multi_attribute_loader_file_list_no_bg,
        "multi_attribute_loader_file_list_ilab" : multi_attribute_loader_file_list_ilab,
        "multi_attribute_loader_file_list_uiuc" : multi_attribute_loader_file_list_uiuc,
        "multi_attribute_loader_file_list_mnist_rotation" : multi_attribute_loader_file_list_mnist_rotation,
        "multi_attribute_loader_file_list_mnist_rotation_29" : multi_attribute_loader_file_list_mnist_rotation_29,
        "multi_attribute_loader_file_list_shapenet" : multi_attribute_loader_file_list_shapenet,
        "multi_attribute_loader_file_list_shapenet_tensor" : multi_attribute_loader_file_list_shapenet_tensor,
        "PennFudanDetection" : PennFudanDataset
    }[name]



def get_loader_classes(name):
    """get_loader_classes
    :param name:
    """
    return {
        "synthetic_binary_segmentation": category_to_class_number_synthetic,
        "synthetic_multi_category_segmentation": category_to_class_number_synthetic,
        "old_dataset_segmentation" : ImageFolder_OLD_DATASET,
        "segmentation_with_paths" : ImageFolder_with_paths,
        "cityscapes" : category_to_class_number_cityscapes,
        "synthetic_multi_category_per_car_segmentation":category_to_class_number_per_car_synthetic,
        "synthetic_multi_category_per_car_segmentation_only_cars":category_to_class_number_per_car_synthetic_only_cars,
        "classification_per_car_model":category_to_class_number_per_car_synthetic_only_cars,
        "multi_attribute_loader":category_to_class_number_per_car_synthetic_only_cars,
        "multi_attribute_loader_file_list":category_to_class_number_per_car_synthetic_only_cars_no_police,
        "multi_attribute_loader_file_list_no_bg":category_to_class_number_per_car_synthetic_only_cars_no_police,
        "multi_attribute_loader_file_list_ilab":category_to_class_number_ilab_5_cars,
        "multi_attribute_loader_file_list_uiuc":category_to_class_number_ilab_5_cars,
        "multi_attribute_loader_file_list_mnist_rotation":category_to_class_number_ilab_5_cars,
        "multi_attribute_loader_file_list_mnist_rotation_29":category_to_class_number_ilab_5_cars,
        "multi_attribute_loader_file_list_shapenet":category_to_class_number_ilab_5_cars,
        "multi_attribute_loader_file_list_shapenet_tensor":category_to_class_number_ilab_5_cars,
        "classification_per_car_model_noisy_texture":category_to_class_number_per_car_synthetic_only_cars,
        "classification_per_car_model_edges_texture":category_to_class_number_per_car_synthetic_only_cars
    }[name]
