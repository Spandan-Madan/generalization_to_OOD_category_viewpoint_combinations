import torch
import torchvision
import torch.nn as nn
from torchvision import models

import torch
import torchvision
import torch.nn as nn
from torchvision import models

class Small_Net(nn.Module):

    def __init__(self, n_class=21):
        super(Small_Net, self).__init__()
        # conv1
        
#         resnet = models.resnet50(pretrained=False)
        conv1 = nn.Conv2d(3,256,kernel_size=8,stride=3,padding=50)

        pool1 = nn.MaxPool2d(kernel_size=2)

        conv2 = nn.Conv2d(256,512,kernel_size=2,stride=1,padding=1)

        pool2 = nn.MaxPool2d(kernel_size=2)


        conv3 = nn.Conv2d(512,4096,kernel_size=4,stride=3)
#         conv1 = nn.Conv2d(3,256,kernel_size=7,stride=2,padding=1)
#         pool1 = nn.MaxPool2d(kernel_size=2)
        
#         conv2 = nn.Conv2d(256,512,kernel_size=3,stride=2,padding=1)
#         pool2 = nn.MaxPool2d(kernel_size=3)
        
#         conv3 = nn.Conv2d(512,4096,kernel_size=2)
        
        self.conv1 = conv1
        self.pool1 = pool1
        self.conv2 = conv2
        self.pool2 = pool2
        self.conv3 = conv3
        
        
#         resnet_beginning = nn.Sequential(*list(resnet.children())[:4])

#         layer1 = list(resnet.children())[4]
#         layer2 = list(resnet.children())[5]
#         layer3 = list(resnet.children())[6]
#         layer4 = list(resnet.children())[7]

#         fc = nn.Sequential(nn.Conv2d(2048,4096,7), nn.ReLU(inplace=True), nn.Dropout2d())
        
#         self.resnet_beginning = resnet_beginning
#         self.layer1 = layer1
#         self.layer2 = layer2
#         self.layer3 = layer3
#         self.layer4 = layer4
#         self.fc = fc
        
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(256, n_class, 1)
        self.score_pool4 = nn.Conv2d(512, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(method='kaiming', *models):
        for model in models:
            for module in model.modules():

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                    if method == 'kaiming':
                        init.kaiming_normal(module.weight.data, np.sqrt(2.0))
                    elif method == 'xavier':
                        init.xavier_normal(module.weight.data, np.sqrt(2.0))
                    elif method == 'orthogonal':
                        init.orthogonal(module.weight.data, np.sqrt(2.0))
                    elif method == 'normal':
                        init.normal(module.weight.data,mean=0, std=0.02)
                    if module.bias is not None:
                        init.constant(module.bias.data,0)

    def forward(self, x):
        h = x
        print('Input')
        print(h.shape)
        
        
        
#         h = self.resnet_beginning(h)
#         print('resnet_beginning')
#         print(h.shape)
        
#         h = self.layer1(h)
        
#         print('layer1')
#         print(h.shape)
        
#         h = self.layer2(h)
        h = self.pool1(self.conv1(h)) 
        layer2 = h
        pool3 = layer2
        print('layer2')
        print(h.shape)
        
#         h = self.layer3(h)
        h = self.pool2(self.conv2(h))
        layer3 = h
        pool4 = layer3
        print('layer3')
        print(h.shape)
        
#         h = self.layer4(h)
#         print('layer4')
#         print(h.shape)
        
#         h = self.fc(h)
        h = self.conv3(h)
        print('fc')
        print(h.shape)
        
        h = self.score_fr(h)
        print('score_fr')
        print(h.shape)
        
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        print('upscore2')
        print(h.shape)
        
        h = self.score_pool4(pool4)
        print('score_pool4')
        print(h.shape)
        
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        print('score_pool4c')
        print(score_pool4c.shape)
        
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        print('upscore_pool4')
        print(upscore_pool4.shape)
        
        h = self.score_pool3(pool3)
        print('score_pool3')
        print(h.shape)
        
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        print('score_pool3c')
        print(score_pool3c.shape)

        h = upscore_pool4 + score_pool3c  # 1/8
        
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        print('upscore8')
        print(h.shape)
        return h

class ResNet_50_FCN8s(nn.Module):

    def __init__(self, n_class=21):
        super(ResNet_50_FCN8s, self).__init__()
        # conv1
        
        resnet = models.resnet50(pretrained=False)

        resnet.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=100)

        resnet_beginning = nn.Sequential(*list(resnet.children())[:4])

        layer1 = list(resnet.children())[4]
        layer2 = list(resnet.children())[5]
        layer3 = list(resnet.children())[6]
        layer4 = list(resnet.children())[7]

        fc = nn.Sequential(nn.Conv2d(2048,4096,7), nn.ReLU(inplace=True), nn.Dropout2d())
        
        self.resnet_beginning = resnet_beginning
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.fc = fc
        
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(512, n_class, 1)
        self.score_pool4 = nn.Conv2d(1024, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(method='kaiming', *models):
        for model in models:
            for module in model.modules():

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                    if method == 'kaiming':
                        init.kaiming_normal(module.weight.data, np.sqrt(2.0))
                    elif method == 'xavier':
                        init.xavier_normal(module.weight.data, np.sqrt(2.0))
                    elif method == 'orthogonal':
                        init.orthogonal(module.weight.data, np.sqrt(2.0))
                    elif method == 'normal':
                        init.normal(module.weight.data,mean=0, std=0.02)
                    if module.bias is not None:
                        init.constant(module.bias.data,0)

    def forward(self, x):
        h = x
        print('Input')
        print(h.shape)
        
        h = self.resnet_beginning(h)
        print('resnet_beginning')
        print(h.shape)
        
        h = self.layer1(h)
        
        print('layer1')
        print(h.shape)
        
        h = self.layer2(h)
        layer2 = h
        pool3 = layer2
        print('layer2')
        print(h.shape)
        
        h = self.layer3(h)
        layer3 = h
        pool4 = layer3
        print('layer3')
        print(h.shape)
        
        h = self.layer4(h)
        print('layer4')
        print(h.shape)
        
        h = self.fc(h)
        print('fc')
        print(h.shape)
        
        h = self.score_fr(h)
        print('score_fr')
        print(h.shape)
        
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        print('upscore2')
        print(h.shape)
        
        h = self.score_pool4(pool4)
        print('score_pool4')
        print(h.shape)
        
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        print('score_pool4c')
        print(score_pool4c.shape)
        
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        print('upscore_pool4')
        print(upscore_pool4.shape)
        
        h = self.score_pool3(pool3)
        print('score_pool3')
        print(h.shape)
        
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        print('score_pool3c')
        print(score_pool3c.shape)

        h = upscore_pool4 + score_pool3c  # 1/8
        
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        print('upscore8')
        print(h.shape)
        return h

class ResNet_152_FCN8s(nn.Module):

    def __init__(self, n_class=21):
        super(ResNet_152_FCN8s, self).__init__()
        # conv1
        
        resnet = models.resnet152(pretrained=False)

        resnet.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=100)

        resnet_beginning = nn.Sequential(*list(resnet.children())[:4])

        layer1 = list(resnet.children())[4]
        layer2 = list(resnet.children())[5]
        layer3 = list(resnet.children())[6]
        layer4 = list(resnet.children())[7]

        fc = nn.Sequential(nn.Conv2d(2048,4096,7), nn.ReLU(inplace=True), nn.Dropout2d())
        
        self.resnet_beginning = resnet_beginning
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.fc = fc
        
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(512, n_class, 1)
        self.score_pool4 = nn.Conv2d(1024, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(method='kaiming', *models):
        for model in models:
            for module in model.modules():

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                    if method == 'kaiming':
                        init.kaiming_normal(module.weight.data, np.sqrt(2.0))
                    elif method == 'xavier':
                        init.xavier_normal(module.weight.data, np.sqrt(2.0))
                    elif method == 'orthogonal':
                        init.orthogonal(module.weight.data, np.sqrt(2.0))
                    elif method == 'normal':
                        init.normal(module.weight.data,mean=0, std=0.02)
                    if module.bias is not None:
                        init.constant(module.bias.data,0)

    def forward(self, x):
        h = x
        print('Input')
        print(h.shape)
        
        h = self.resnet_beginning(h)
        print('resnet_beginning')
        print(h.shape)
        
        h = self.layer1(h)
        
        print('layer1')
        print(h.shape)
        
        h = self.layer2(h)
        layer2 = h
        pool3 = layer2
        print('layer2')
        print(h.shape)
        
        h = self.layer3(h)
        layer3 = h
        pool4 = layer3
        print('layer3')
        print(h.shape)
        
        h = self.layer4(h)
        print('layer4')
        print(h.shape)
        
        h = self.fc(h)
        print('fc')
        print(h.shape)
        
        h = self.score_fr(h)
        print('score_fr')
        print(h.shape)
        
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        print('upscore2')
        print(h.shape)
        
        h = self.score_pool4(pool4)
        print('score_pool4')
        print(h.shape)
        
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        print('score_pool4c')
        print(score_pool4c.shape)
        
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        print('upscore_pool4')
        print(upscore_pool4.shape)
        
        h = self.score_pool3(pool3)
        print('score_pool3')
        print(h.shape)
        
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        print('score_pool3c')
        print(score_pool3c.shape)

        h = upscore_pool4 + score_pool3c  # 1/8
        
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        print('upscore8')
        print(h.shape)
        return h

class ResNet_FCN8s_dropped_layer_2_and_4(nn.Module):

    def __init__(self, n_class=21):
        super(ResNet_FCN8s_dropped_layer_2_and_4, self).__init__()
        # conv1
        
        resnet = models.resnet18(pretrained=False)
        resnet.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=100)

        resnet_beginning = nn.Sequential(*list(resnet.children())[:4])

        layer1 = list(resnet.children())[4]
        layer2 = list(resnet.children())[5]
        layer3 = list(resnet.children())[6]
        layer4 = list(resnet.children())[7]

        fc = nn.Sequential(nn.Conv2d(256,4096,20), nn.ReLU(inplace=True), nn.Dropout2d())
        
        self.resnet_beginning = resnet_beginning
#         self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
#         self.layer4 = layer4
        self.fc = fc
        
        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.score_pool3 = nn.Conv2d(128, n_class, 1)
        self.score_pool4 = nn.Conv2d(256, n_class, 1)

        self.upscore2 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            n_class, n_class, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_class, n_class, 4, stride=2, bias=False)

        self._initialize_weights()

    def _initialize_weights(method='kaiming', *models):
        for model in models:
            for module in model.modules():

                if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
                    if method == 'kaiming':
                        init.kaiming_normal(module.weight.data, np.sqrt(2.0))
                    elif method == 'xavier':
                        init.xavier_normal(module.weight.data, np.sqrt(2.0))
                    elif method == 'orthogonal':
                        init.orthogonal(module.weight.data, np.sqrt(2.0))
                    elif method == 'normal':
                        init.normal(module.weight.data,mean=0, std=0.02)
                    if module.bias is not None:
                        init.constant(module.bias.data,0)

    def forward(self, x):
        h = x
        print('Input')
        print(h.shape)
        
        h = self.resnet_beginning(h)
        print('resnet_beginning')
        print(h.shape)
        
        h = self.layer2(h)
        layer2 = h
        pool3 = layer2
        print('layer2')
        print(h.shape)
        
        h = self.layer3(h)
        layer3 = h
        pool4 = layer3
        print('layer3')
        print(h.shape)

        h = self.fc(h)
        print('fc')
        print(h.shape)
        
        h = self.score_fr(h)
        print('score_fr')
        print(h.shape)
        
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        print('upscore2')
        print(h.shape)
        
        h = self.score_pool4(pool4)
        print('score_pool4')
        print(h.shape)
        
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        print('score_pool4c')
        print(score_pool4c.shape)
        
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        print('upscore_pool4')
        print(upscore_pool4.shape)
        
        h = self.score_pool3(pool3)
        print('score_pool3')
        print(h.shape)
        
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8
        print('score_pool3c')
        print(score_pool3c.shape)

        h = upscore_pool4 + score_pool3c  # 1/8
        
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        print('upscore8')
        print(h.shape)
        return h