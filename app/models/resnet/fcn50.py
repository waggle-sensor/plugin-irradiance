from torchvision.models.segmentation import fcn_resnet50
import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, n_class=21):

        super(FCN, self).__init__()
        self.fcn = fcn_resnet50(pretrained=False, num_classes=n_class)

    def forward(self, x, debug=False):
        return self.fcn(x)['out']

