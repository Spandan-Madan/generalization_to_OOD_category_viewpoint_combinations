import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image


# In[2]:


# def initialize_weights(method='kaiming', *models):
#     for model in models:
#         for module in model.modules():

#             if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
#                 if method == 'kaiming':
#                     init.kaiming_normal(module.weight.data, np.sqrt(2.0))
#                 elif method == 'xavier':
#                     init.xavier_normal(module.weight.data, np.sqrt(2.0))
#                 elif method == 'orthogonal':
#                     init.orthogonal(module.weight.data, np.sqrt(2.0))
#                 elif method == 'normal':
#                     init.normal(module.weight.data,mean=0, std=0.02)
#                 if module.bias is not None:
#                     init.constant(module.bias.data,0)


# In[8]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UnetEncoder(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(UnetEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1),
                              nn.BatchNorm2d(self.out_channels),
                              nn.ReLU(),
                              nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
                              nn.BatchNorm2d(self.out_channels),
                              nn.ReLU())

    def forward(self,x):
        return self.layer(x)


class UnetDecoder(nn.Module):

    def __init__(self, in_channels, block_features, out_channels):
        super(UnetDecoder, self).__init__()
        self.in_channels = in_channels
        self.features = block_features
        self.out_channels = out_channels

        self.layer = nn.Sequential(nn.Conv2d(self.in_channels, self.features, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.features),
                                   nn.ReLU(),
                                   nn.Conv2d(self.features, self.features, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.features),
                                   nn.ReLU(),
                                   nn.ConvTranspose2d(self.features, self.out_channels, kernel_size=2, stride=2),
                                   nn.BatchNorm2d(self.out_channels),
                                   nn.ReLU())

    def forward(self,x):
        return self.layer(x)

class UNet(nn.Module):
    def __init__(self, num_classes):
        print('initializing unet spandan')
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.down1 = UnetEncoder(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down2 = UnetEncoder(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down3 = UnetEncoder(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.down4 = UnetEncoder(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.center = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                                    nn.BatchNorm2d(1024),
                                    nn.ReLU(),
                                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
                                    nn.BatchNorm2d(512),
                                    nn.ReLU())

        self.up1 = UnetDecoder(1024, 512, 256)
        self.up2 = UnetDecoder(512, 256, 128)
        self.up3 = UnetDecoder(256, 128, 64)

        self.up4 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(),
                                 nn.Conv2d(64, 64, 3, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU())

        self.output = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1)
        self.normalizing_layer = nn.Softmax(1)


        #Initialize weights
#         initialize_weights(self)


    def forward(self, x):
        
        # Down Block
        down_1_out = self.down1(x)
        pool_1_out = self.pool1(down_1_out)
        
        down_2_out = self.down2(pool_1_out)
        pool_2_out = self.pool2(down_2_out)
        
        down_3_out = self.down3(pool_2_out)
        pool_3_out = self.pool3(down_3_out)
        
        down_4_out = self.down4(pool_3_out)
        pool_4_out = self.pool4(down_4_out)
        
        
        # Center Block
        center_right = self.center(pool_4_out)
        center_left = down_4_out
        concat_1 = torch.cat([center_left,center_right],1)
        
        up_1_out = self.up1(concat_1)
        
        up_1_right = up_1_out
        up_1_left = down_3_out
        concat_2 = torch.cat([up_1_left,up_1_right],1)
        
        up_2_out = self.up2(concat_2)
        
        up_2_right = up_2_out
        up_2_left = down_2_out
        concat_3 = torch.cat([up_2_left,up_2_right],1)
        
        up_3_out = self.up3(concat_3)
        
        up_3_right = up_3_out
        up_3_left = down_1_out
        concat_4 = torch.cat([up_3_left,up_3_right],1)
        
        up_4_out = self.up4(concat_4)
        
        seg_map = self.output(up_4_out)
        
        normalized_map = self.normalizing_layer(seg_map)
        out = normalized_map

        return out



#################################UNCOMMENT CODE BELOW TO RUN TESTING #######################
# unet = UNet(2)


# # In[11]:


# input_impath = '/data/graphics/toyota-pytorch/viewpoint_data_png/infinity_jx2013/0_8_pos_00_angle_000.png'
# im = Image.open(input_impath)
# im_resize = im.resize((32,32))
# imarr = np.array(im_resize)
# imarr = imarr[:,:,:-1]
# t = torch.from_numpy(imarr)
# t2 = t.permute([2,0,1]).unsqueeze(0).float()


# # In[14]:


# out = unet(t2)

