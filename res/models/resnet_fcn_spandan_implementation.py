import torch
import torchvision
import torch.nn as nn
from torchvision import models

class ResNet_FCN8s(nn.Module):

    def __init__(self, n_class=21):
        super(ResNet_FCN8s, self).__init__()
        # conv1
        
        resnet = models.resnet18(pretrained=False)

        resnet.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=100)

        resnet_beginning = nn.Sequential(*list(resnet.children())[:4])

        layer1 = list(resnet.children())[4]
        layer2 = list(resnet.children())[5]
        layer3 = list(resnet.children())[6]
        layer4 = list(resnet.children())[7]
        fc = nn.Sequential(nn.Conv2d(512,4096,7), nn.ReLU(inplace=True), nn.Dropout2d())
        
        self.resnet_beginning = resnet_beginning
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
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
        
        #self.final = nn.Sigmoid()
        self.final = nn.Softmax(1)

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
        h = self.resnet_beginning(h)
        h = self.layer1(h)
        
        h = self.layer2(h)

        layer2 = h
        pool3 = layer2
        h = self.layer3(h)
        layer3 = h
        pool4 = layer3
       
        h = self.layer4(h)
         
        h = self.fc(h)
        h = self.score_fr(h)
        
        h = self.upscore2(h)
        upscore2 = h  # 1/16
        
        h = self.score_pool4(pool4)
        
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16
        
        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8
        
        h = self.score_pool3(pool3)
        
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8
        
        h = self.upscore8(h)
        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        
        return self.final(h)
