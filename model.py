import torch
import torchvision.models as tmodels
import torch.nn as nn
import os

class ModelResNet18(nn.Module):
    def __init__(self, pretrained=True):
        super(ModelResNet18, self).__init__()
        self.backbone = tmodels.resnet18(pretrained=pretrained)
        self.convolution2d = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
        self.fc_avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x) 
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.convolution2d(x)
        x = self.fc_avgpool(x)
        
        x = torch.flatten(x, 1)
        return x
    
    
    def train_update(self, train_tensor_tuple, criterion):
        xb, yb = train_tensor_tuple

        outputs = self.forward(xb)
        loss = criterion(outputs, yb) 
        
        return loss

    def predict(self, item, is_prob = False):
        xb, _ = item 
           
        xb = xb.to(os.environ['PU'])
        with torch.no_grad():
            outputs = self.forward(xb)
            if is_prob:
                outputs = torch.sigmoid(outputs)
        return outputs
