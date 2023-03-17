import torch
import torch.nn as nn
import torchvision.models

# A class for network: the class defines convolutional neural network (CNN) 
class ItemMatchEfficientNet(nn.Module):
  def __init__(self,type,pretrained=True):
    super().__init__()
    self.type = type
    self.pretrained = pretrained

    if self.type == 'b0':
      self.backbone = torchvision.models.efficientnet_b0(pretrained==self.pretrained)

    
    if self.type == 'b7':
      self.backbone = torchvision.models.efficientnet_b7(pretrained=self.pretrained)

    lastInput = self.backbone.classifier[1].in_features
    self.backbone.classifier[1]= nn.Linear(in_features = lastInput, out_features=8000)

    # define feedforward layer after CNN backbone
    self.fc1 = nn.Linear(8000,16000) # the feedforward layer
    self.act = nn.ReLU()
    # define a dropout
    self.dropout = nn.Dropout(0.5) # dropout rate 
    self.fc2 = nn.Linear(16000,5) 


  def forward(self,x):
      x = self.backbone(x)
      x =self.fc1(x)
      x = self.act(x)
      x = self.dropout(x)
      output = self.fc2(x)
      return output

# Customizing efficientNet for P3 Identify Diabetic Retina project
class DiabeticEfficientNet(nn.Module):
    def __init__(self, depth, pretrained=True, num_classes=None):
        super(DiabeticEfficientNet,self).__init__()
        self.depth = depth
        self.pretrained = pretrained
        self.num_classes = num_classes

        if self.depth == 'b0':
            self.backbone = torchvision.models.efficientnet_b0(pretrained = self.pretrained)

        if self.depth == 'b1':
            self.backbone = torchvision.models.efficientnet_b1(pretrained = self.pretrained)

        if self.depth == 'b2':
            self.backbone = torchvision.models.efficientnet_b2(pretrained = self.pretrained)

        if self.depth == 'b3':
            self.backbone = torchvision.models.efficientnet_b3(pretrained = self.pretrained)

        if self.depth == 'b4':
            self.backbone = torchvision.models.efficientnet_b4(pretrained = self.pretrained)

        if self.depth == 'b5':
            self.backbone = torchvision.models.efficientnet_b5(pretrained = self.pretrained)

        if self.depth == 'b6':
            self.backbone = torchvision.models.efficientnet_b6(pretrained = self.pretrained)

        if self.depth == 'b7':
            self.backbone = torchvision.models.efficientnet_b7(pretrained = self.pretrained)

        if self.num_classes != None:
            lastInput = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features=lastInput, out_features=self.num_classes)

        # # define feedforward layer after CNN backbone
        # self.fc1 = nn.Linear(8000,16000) # the feedforward layer
        # self.act = nn.ReLU()
        # # define a dropout
        # self.dropout = nn.Dropout(0.5) # dropout rate
        # self.fc2 = nn.Linear(16000,num_classes)

    def forward(self, x):
        output = self.backbone(x)
        # x =self.fc1(x)
        # x = self.act(x)
        # x = self.dropout(x)
        # output = self.fc2(x)
        return output


# Customizing Renset nns for P3 Identify Diabetic Retina project
class DiabeticResNet(nn.Module):
    def __init__(self, depth, pretrained=True, num_classes=None):
        super(DiabeticResNet, self).__init__()
        self.depth = depth
        self.pretrained = pretrained
        self.num_classes = num_classes

        if self.depth == 50:
            self.backbone = torchvision.models.resnet50(pretrained=self.pretrained)

        if self.depth == 18:
            self.backbone = torchvision.models.resnet18(pretrained=self.pretrained)

        if self.num_classes != None:
            lastInput = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features=lastInput, out_features=self.num_classes)

    def forward(self, x):
        output = self.backbone(x)
        return output

# Customizing Densenet for P3 Identify Diabetic Retina project
class DiabeticDenseNet(nn.Module):
    class Flatten(nn.Module):
      def __init__(self):
        super(DiabeticDenseNet.Flatten, self).__init__()

      def forward(self, x):
        out = nn.functional.relu(x, inplace=True)
        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        return out

    def __init__(self, depth, pretrained=True, drop_classifier = False, **kwargs):
        super(DiabeticDenseNet,self).__init__()
        self.depth = depth
        self.pretrained = pretrained

        if self.depth == 'densenet121':
            self.backbone = torchvision.models.densenet121(pretrained = self.pretrained, **kwargs)

        if self.depth == 'densenet161':
            self.backbone = torchvision.models.densenet161(pretrained = self.pretrained, **kwargs)

        if self.depth == 'densenet169':
            self.backbone = torchvision.models.densenet169(pretrained = self.pretrained, **kwargs)

        if self.depth == 'densenet201':
            self.backbone = torchvision.models.densenet201(pretrained = self.pretrained, **kwargs)

        # picking the number of input features for the last layer of the classsifier
        self.in_features = self.backbone.classifier.in_features
        if drop_classifier:
          # selecting the pretrained model without the last classifier part
          self.backbone  = nn.Sequential(*list(self.backbone.children())[:-1], DiabeticDenseNet.Flatten())

    def forward(self, x):
        output = self.backbone(x)
        return output

# Ensembling two denseNet for P3 Identify Diabetic Retina project
class DiabeticEnsembleDenseNets(nn.Module):
    # class Flatten(nn.Model):
    #   def __init
    def __init__(self, model1, model2, num_classes=None):
        super(DiabeticEnsembleDenseNets, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.num_classes = num_classes

        if self.num_classes != None: 
          # adding the fully connected layer
            self.fc = nn.Linear(self.model1.in_features + self.model2.in_features, self.num_classes)

    def forward(self, x):
        x1 = self.model1.backbone(x)

        # out = nn.functional.relu(features, inplace=True)
        # out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
        # out = torch.flatten(out, 1)



        x2 = self.model2.backbone(x)
        # x1 = torch.flatten(x1, 1)
        # x2 = torch.flatten(x2, 1)
        output = torch.cat((x1, x2), dim=1)
        # print(f'x1 size {x1.{output.size()}, x2 size {x2.size()}')
        # print(f'output size size()}')
        output = output.view(x.size(0), -1)
        # print(f'output size 2 {output.size()}, x size {x.size()}')

        output = self.fc(output)
        return output

# Ensembling two efficientNets for P3 Identify Diabetic Retina project
class DiabeticEnsembleEfficientNets(nn.Module):
    def __init__(self, model1, model2, num_classes=None):
        super(DiabeticEnsembleEfficientNets, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.num_classes = num_classes

        if self.num_classes != None: 
          # picking the number of input features for the last layer of the classsifier
            Input1 = self.model1.backbone.classifier[-1].in_features
            Input2 = self.model2.backbone.classifier[-1].in_features
          # selecting the pretrained model without the last classifier part
            self.m1_layers = nn.Sequential(*list(self.model1.backbone.children())[:-1])
            self.m2_layers = nn.Sequential(*list(self.model2.backbone.children())[:-1])
          # adding the fully connected layer
            self.fc = nn.Linear(Input1 + Input2, self.num_classes)

    def forward(self, x):
        x1 = self.m1_layers(x)
        x2 = self.m2_layers(x)
        output = torch.cat((x1, x2), dim=1)
        print(f'x1 size {x1.size()}, x2 size {x2.size()}')
        print(f'output size {output.size()}')
        output = output.view(x.size(0), -1)
        print(f'output size 2 {output.size()}, x size {x.size()}')

        output = self.fc(output)
        return output


