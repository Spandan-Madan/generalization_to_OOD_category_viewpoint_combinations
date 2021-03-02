import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class MultiAttributeLoss(torch.nn.Module):
    def __init__(self):
        super(MultiAttributeLoss, self).__init__()
#         self.closs = nn.CrossEntropyLoss()
#         self.count = count

    def forward(self, input, target):
        product = 1
        count = len(input)
        for i in range(count):
            attribute_loss = F.cross_entropy(input[i],target[i])
            product *= attribute_loss
        
        geometric_mean = torch.pow(product,count)
        return geometric_mean    
#         return F.cross_entropy(input, target, weight=self.weight,
#                                ignore_index=self.ignore_index, reduction=self.reduction)



weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.1]
weights_2 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0]

class_weights = torch.FloatTensor(weights).cuda()
class_weights_2 = torch.FloatTensor(weights_2).cuda()

loss_list = { 'crossentropy': nn.CrossEntropyLoss(),'weighted_crossentropy':nn.CrossEntropyLoss(weight=class_weights),'crossentropy_no_bg':nn.CrossEntropyLoss(weight=class_weights_2),'multi_attribute_loss':MultiAttributeLoss()}


def get_loss(loss):
    return loss_list[loss]
