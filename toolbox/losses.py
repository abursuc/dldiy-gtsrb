'''
Custom loss functions
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_criterion(args, class_weights=None):
    
    if args.class_balance is None:
        class_weights = None
        return {
            'crossentropy': nn.CrossEntropyLoss(ignore_index=255),
            'nll': nn.NLLLoss(ignore_index=255),
            'bce': nn.BCELoss(),
            'focal': FocalLoss(ignore_index=255),
            'mse': nn.MSELoss(reduction='elementwise_mean'), 
            'l1': nn.L1Loss(reduction='elementwise_mean'), 
        }[args.criterion]
        
    else:
        return {
            'focal': FocalLoss(weight=class_weights.float(), ignore_index=255),
            'nll': nn.NLLLoss(weight=class_weights.float(), ignore_index=255),
        }[args.criterion]
        

''' 
 oooo
 `888
  888   .ooooo.   .oooo.o  .oooo.o  .ooooo.   .oooo.o
  888  d88' `88b d88(  "8 d88(  "8 d88' `88b d88(  "8
  888  888   888 `"Y88b.  `"Y88b.  888ooo888 `"Y88b.
  888  888   888 o.  )88b o.  )88b 888    .o o.  )88b
 o888o `Y8bod8P' 8""888P' 8""888P' `Y8bod8P' 8""888P'
''' 


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss((1 - F.softmax(inputs,dim=0)) ** self.gamma * F.log_softmax(inputs,dim=0), targets)



class GramMatrix(nn.Module):
    def __init__(self):
        super(GramMatrix, self).__init__()

    def forward(self, input):
        features = input.view(input.shape[0], input.shape[1], -1)
        gram_matrix = torch.bmm(features, features.transpose(1,2))
        return gram_matrix

