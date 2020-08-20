#from torchvision.models.segmentation import fcn_resnet101
import torch
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, n_class=21):

        super(FCN, self).__init__()
        #self.fcn = fcn_resnet101(pretrained=False, num_classes=n_class)
        # Uses bilinear interpolation for upsampling
        # https://github.com/pytorch/vision/blob/master/
        # torchvision/models/segmentation/_utils.py
        checkpoint = torch.load('resnet101-5d3b4d8f.pth')
        self.fcn = model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x, debug=False):
        return self.fcn(x)['out']


