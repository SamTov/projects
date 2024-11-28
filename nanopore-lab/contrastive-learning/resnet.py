import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit


def format_params_in_millions(param_count):
    return f"{param_count / 1e6:.2f}M"


def count_parameters(model):
    return format_params_in_millions(sum(p.numel() for p in model.parameters() if p.requires_grad))


class BasicBlock(jit.ScriptModule):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    @jit.script_method
    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class Bottleneck(jit.ScriptModule):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.downsample = downsample
        self.stride = stride

    @jit.script_method
    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out

class CustomResNet(jit.ScriptModule):
    def __init__(
            self,
            block, 
            layers, 
            num_classes=1000, 
            in_channels=2, 
            scale_factor: int = 1
        ):
        super(CustomResNet, self).__init__()
        self.in_channels = 64 * scale_factor

        self.conv1 = nn.Conv2d(
            in_channels, 
            64 * scale_factor,
            kernel_size=7, 
            stride=2, 
            padding=3, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64 * scale_factor)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64 * scale_factor, layers[0])
        self.layer2 = self._make_layer(block, 128 * scale_factor, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256 * scale_factor, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512 * scale_factor, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(scale_factor * 512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    
    @jit.script_method
    def latent_space(self, x):
        """
        Compute the latent space of the model.
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    @jit.script_method
    def forward(self, x):
        x = self.latent_space(x)
        x = self.fc(x)

        return x

def resnet18(num_classes, in_channels, scale_factor):
    return CustomResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels, scale_factor)

def resnet34(num_classes, in_channels, scale_factor):
    return CustomResNet(BasicBlock, [3, 4, 6, 3], num_classes, in_channels, scale_factor)

def resnet50(num_classes, in_channels, scale_factor):
    return CustomResNet(Bottleneck, [3, 4, 6, 3], num_classes, in_channels, scale_factor)

def resnet101(num_classes, in_channels, scale_factor):
    return CustomResNet(Bottleneck, [3, 4, 23, 3], num_classes, in_channels, scale_factor)

def resnet152(num_classes, in_channels, scale_factor):
    return CustomResNet(Bottleneck, [3, 8, 36, 3], num_classes, in_channels, scale_factor)

# # Example usage
# resnet_size = 152  # Choose from [18, 34, 50, 101, 152]
# num_classes = 42  # Number of output classes
# in_channels = 2   # Number of input channels

# if resnet_size == 18:
#     model = resnet18(num_classes, in_channels)
# elif resnet_size == 34:
#     model = resnet34(num_classes, in_channels)
# elif resnet_size == 50:
#     model = resnet50(num_classes, in_channels)
# elif resnet_size == 101:
#     model = resnet101(num_classes, in_channels)
# elif resnet_size == 152:
#     model = resnet152(num_classes, in_channels)
# else:
#     raise ValueError("Invalid resnet_size value. Choose from [18, 34, 50, 101, 152].")

