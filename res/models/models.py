import sys
import os
sys.path.append('/data/graphics/toyota-pytorch/training_scaffold_own/res/')
import torchvision
# print(__file__)
# models_folder = '/'.join(__file__.split('/')[:-1])

# sys.path.append(models_folder)
print(os.getcwd())
# print(models)
def get_model(MODEL_ARCH,NUM_CLASSES):
    if MODEL_ARCH == 'PSPNET':
        from models.pspnet_model import PSPNet
        return PSPNet(num_classes=NUM_CLASSES,pretrained=False,use_aux=False)
    if MODEL_ARCH == 'UNET':
        from models.Unet_spandan_implementation import UNet
        return UNet(num_classes=NUM_CLASSES)
    if MODEL_ARCH == 'VGGFCN':
        from models.FCN import FCN8s
        return FCN8s(n_class=NUM_CLASSES)
    if MODEL_ARCH == 'RESNETFCN':
        from models.resnet_fcn_spandan_implementation import ResNet_FCN8s
        return ResNet_FCN8s(n_class=NUM_CLASSES)
    if MODEL_ARCH == 'DEEPLAB':
        from models import deeplab
        return deeplab.DeepLab(backbone='resnet',num_classes=NUM_CLASSES)
    if MODEL_ARCH == 'SEGNET':
        from models.segnet import SegNet
        return SegNet(input_channels=3,output_channels=NUM_CLASSES)
    if MODEL_ARCH == 'R2UNET':
        from models.r2unet import R2AttU_Net
        return R2AttU_Net(img_ch=3,output_ch=NUM_CLASSES)
    if MODEL_ARCH == 'SMALLNET':
        from models.resnet_fcn_spandan_implementation_modified import Small_Net
        return Small_Net(n_class=NUM_CLASSES)
    if MODEL_ARCH == 'RESNETSMALL':
        from models.resnet_fcn_spandan_implementation_modified import ResNet_FCN8s_dropped_layer_2_and_4
        return ResNet_FCN8s_dropped_layer_2_and_4(n_class=NUM_CLASSES)
    if MODEL_ARCH == 'RESNET50':
        from models.resnet_fcn_spandan_implementation_modified import ResNet_50_FCN8s
        return ResNet_50_FCN8s(n_class=NUM_CLASSES)
    if MODEL_ARCH == 'RESNET152':
        from models.resnet_fcn_spandan_implementation_modified import ResNet_152_FCN8s
        return ResNet_152_FCN8s(n_class=NUM_CLASSES)
    if MODEL_ARCH == 'RESNETFCN050':
        from models.resnet_channel_variants import ResNet18_channel_modified
        return ResNet18_channel_modified([32,64,128,256], n_class=NUM_CLASSES)
    if MODEL_ARCH == 'RESNETFCN025':
        from models.resnet_channel_variants import ResNet18_channel_modified
        return ResNet18_channel_modified([16,32,64,128] ,n_class=NUM_CLASSES)
    if MODEL_ARCH == 'RESNETFCN0125':
        from models.resnet_channel_variants import ResNet18_channel_modified
        return ResNet18_channel_modified([8,16,32,64] ,n_class=NUM_CLASSES)
    if MODEL_ARCH == 'RESNETFCN00625':
        from models.resnet_channel_variants import ResNet18_channel_modified
        return ResNet18_channel_modified([4,8,16,32] ,n_class=NUM_CLASSES)
    if MODEL_ARCH == 'DEEPLABNONATROUS':
        from models import deeplabnonatrous
        return deeplabnonatrous.DeepLab(backbone='resnet',output_stride=1,num_classes=NUM_CLASSES)
    if MODEL_ARCH == 'LENET':
        from models import lenet
        return lenet.LeNet(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'RESNET18':
        import torchvision
        from torchvision.models import resnet18
        import torch.nn as nn
        model_ft = resnet18(pretrained = False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, NUM_CLASSES)
        return model_ft
    if MODEL_ARCH == 'MULTITASKRESNET':
        from models.multitask_resnet import resnet18
#         print(resnet18)
#         import models as mm
#         print(mm.__file__)
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASKRESNET_EARLY_BRANCHING':
        from models.multitask_resnet_early_branching import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'One_Layer_Encoder_Resnet_Decoder':
        from models.One_Layer_Encoder_Resnet_Decoder import resnet18_variant
        return resnet18_variant(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'Split_Resnet_After_One_Conv':
        from models.Split_Resnet_After_One_Conv import resnet18_variant
        return resnet18_variant(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'Multitask_Resnet_Early_New':
        from models.Multitask_Resnet_Early_New import resnet18_variant
        return resnet18_variant(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASKRESNET_NEW':
        from models.multitask_resnet_new import resnet18_variant
        return resnet18_variant(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'Two_Block_Encoder_Long_Decoder':
        from models.Two_Block_Encoder_Long_Decoder import resnet18_variant
        return resnet18_variant(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'Two_Block_Encoder_Long_Decoder':
        from models.Two_Block_Encoder_Long_Decoder import resnet18_variant
        return resnet18_variant(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'Long_Decoder_Long_Encoder':
        from models.Long_Decoder_Long_Encoder import resnet18_variant
        return resnet18_variant(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'EARLY_BRANCHING_COMBINED':
        from models.EARLY_BRANCHING_COMBINED import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'SPLIT_AFTER_ONE_BLOCK':
        from models.SPLIT_AFTER_ONE_BLOCK import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'LATE_BRANCHING_COMBINED':
        from models.LATE_BRANCHING_COMBINED import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'LATE_BRANCHING_COMBINED_WIDER':
        from models.LATE_BRANCHING_COMBINED_WIDER import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'LATE_BRANCHING_COMBINED_HALF':
        from models.LATE_BRANCHING_COMBINED_HALF import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'LATE_BRANCHING_COMBINED_ONE_FOURTH':
        from models.LATE_BRANCHING_COMBINED_ONE_FOURTH import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'LATE_BRANCHING_COMBINED_FOUR_TIMES':
        from models.LATE_BRANCHING_COMBINED_FOUR_TIMES import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'SPLIT_AFTER_THREE_BLOCKS':
        from models.SPLIT_AFTER_THREE_BLOCKS import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'LONG_DECODER_COMBINED':
        from models.LONG_DECODER_COMBINED import resnet18
        return resnet18(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASK_RESNEXT':
        from models.multitask_resnext import resnext50_32x4d
        return resnext50_32x4d(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASK_WIDE_RESNET':
        from models.multitask_resnext import wide_resnet50_2
        return wide_resnet50_2(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASK_INCEPTION':
        from models.multitask_inception import inception_v3
        return inception_v3(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASK_DENSENET':
        from models.multitask_densenet import densenet121
        return densenet121(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASK_DENSENET_WIDE':
        from models.multitask_densenet_wide import densenet121_wide
        return densenet121_wide(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASK_INCEPTION_WIDE':
        from models.multitask_inception_wide import inception_v3
        return inception_v3(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASK_RESNEXT_WIDE':
        from models.LATE_BRANCHING_COMBINED_WIDER import resnext50_32x4d
        return resnext50_32x4d(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MULTITASK_WIDE_RESNET_WIDE':
        from models.LATE_BRANCHING_COMBINED_WIDER import wide_resnet50_2
        return wide_resnet50_2(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'ALEXNET_LEAST_WIDE':
        from models.alexnet_least_wide import AlexNet_Least_Wide
        return AlexNet_Least_Wide(num_classes = NUM_CLASSES)
    if MODEL_ARCH == 'MASKRCNN':
        import torchvision
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           NUM_CLASSES)
        return model
    if MODEL_ARCH == 'VISION_TRANSFORMER':
        from models.vision_transformer import vision_transformer_model
        return vision_transformer_model(num_classes = NUM_CLASSES)

#     if MODEL_ARCH == 'MULTITASK_SQUEEZENET':
#         from models.multitask_squeezenet import squeezenet1_1
#         return squeezenet1_1(num_classes = NUM_CLASSES)
# #
# #
# def get_loader(name):
#     """get_loader
#     :param name:
#     """
#     return {
#         "synthetic_binary_segmentation": ImageFolder,
#         "synthetic_multi_category_segmentation": multi_category_ImageFolder,
#         "old_dataset_segmentation":ImageFolder_OLD_DATASET,
#         "segmentation_with_paths":ImageFolder_with_paths,
#     }[name]
