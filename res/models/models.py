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
