import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = F.relu(self.fc1(inputs))
        tensor = F.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)
        return tensor


class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class Mnist_EX(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        tensor = self.fc2(tensor)
        return tensor


class OneHiddenLayerFc(nn.Module):
    def __init__(self, input_shape, out_dim, mid_dim=100):
        super(OneHiddenLayerFc, self).__init__()
        self.fc1 = nn.Linear(input_shape, mid_dim)
        self.fc2 = nn.Linear(mid_dim, out_dim)

        self.weight_keys = [['fc1.weight', 'fc1.bias'],
                            ['fc2.weight', 'fc2.bias']]

    def forward(self, x):
        x = x.view(-1, 3*32*32)
        mid_features = F.relu(self.fc1(x))
        out = self.fc2(mid_features)
        return out


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Mnist_CNN2(nn.Module):
    def __init__(self):
        super(Mnist_CNN2, self).__init__()
        self.base = FE2().to(dev)
        self.classifier = Classifier2().to(dev)

    def forward(self, x):
        x = x.to(dev)
        return self.classifier((self.base(x)))

    def extract_feature(self, x):
        x = x.to(dev)
        return self.base(x)


class FE2(nn.Module):
    def __init__(self):
        super(FE2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)

    def forward(self, inputs):
        tensor = inputs.view(-1, 1, 28, 28)
        tensor = F.relu(self.conv1(tensor))
        tensor = self.pool1(tensor)
        tensor = F.relu(self.conv2(tensor))
        tensor = self.pool2(tensor)
        tensor = tensor.view(-1, 7 * 7 * 64)
        tensor = F.relu(self.fc1(tensor))
        return tensor


class Classifier2(nn.Module):
    def __init__(self):
        super(Classifier2, self).__init__()
        self.fc2 = nn.Linear(512, 10).to(dev)

    def forward(self, inputs):
        tensor = self.fc2(inputs)
        return tensor


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)

        return F.relu(y + x)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))

    return nn.Sequential(*blk)


class ResNet2(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNet2, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = nn.Sequential(
            resnet_block(64, 64, 2, first_block=True),
            resnet_block(64, 128, 2)
        )
        self.fc = nn.Linear(128, num_classes)

        self.weight_keys = [['block1.0.weight', 'block1.0.bias'],
                            ['block1.1.weight', 'block1.1.bias'],
                            ['block2.0.0.conv1.weight', 'block2.0.0.conv1.bias'],
                            ['block2.0.0.conv2.weight', 'block2.0.0.conv2.bias'],
                            ['block2.0.0.bn1.weight', 'block2.0.0.bn1.bias'],
                            ['block2.0.0.bn2.weight', 'block2.0.0.bn2.bias'],
                            ['block2.0.1.conv1.weight', 'block2.0.1.conv1.bias'],
                            ['block2.0.1.conv2.weight', 'block2.0.1.conv2.bias'],
                            ['block2.0.1.bn1.weight', 'block2.0.1.bn1.bias'],
                            ['block2.0.1.bn2.weight', 'block2.0.1.bn2.bias'],
                            ['block2.1.0.conv1.weight', 'block2.1.0.conv1.bias'],
                            ['block2.1.0.conv2.weight', 'block2.1.0.conv2.bias'],
                            ['block2.1.0.conv3.weight', 'block2.1.0.conv3.bias'],
                            ['block2.1.0.bn1.weight', 'block2.1.0.bn1.bias'],
                            ['block2.1.0.bn2.weight', 'block2.1.0.bn2.bias'],
                            ['block2.1.1.conv1.weight', 'block2.1.1.conv1.bias'],
                            ['block2.1.1.conv2.weight', 'block2.1.1.conv2.bias'],
                            ['block2.1.1.bn1.weight', 'block2.1.1.bn1.bias'],
                            ['block2.1.1.bn2.weight', 'block2.1.1.bn2.bias'],
                            ['fc.weight', 'fc.bias']]


    def forward(self, x):
        if x.size(1) == 784:
            x = x.view(-1, 1, 28, 28)
        if x.ndim == 3:
            x = torch.unsqueeze(x, dim=1)


        y = self.block1(x)


        block2_output = self.block2(y)


        h = block2_output.view(-1, block2_output.shape[1] * block2_output.shape[2] * block2_output.shape[3])


        pooled = F.avg_pool2d(block2_output, kernel_size=block2_output.size()[2:])


        ch = pooled.view(-1, 128)


        x = self.fc(ch)

        return ch, h, x


class FashionMnistNet(nn.Module):
    def __init__(self):
        super(FashionMnistNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 40, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(40, 80, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 1),
            nn.Conv2d(80, 160, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(3, 3), )

        self.classifier = nn.Sequential(
            nn.Linear(160 * 8 * 8, 200),
            nn.ReLU(),


            nn.Dropout(p=0.5),
            nn.Linear(200, 10))

    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, 1, 28, 28)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


def init_kaiming(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


class CNN_mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = torch.nn.Dropout()
        self.fc1 = torch.nn.Linear(320, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

        pass

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        h = F.relu(F.max_pool2d(self.conv1(x), 2))
        h = F.relu(F.max_pool2d(self.dropout(self.conv2(h)), 2))
        h = h.view(-1, h.shape[1] * h.shape[2] * h.shape[3])
        ch = F.relu(self.fc1(h))
        ch = self.fc2(ch)
        x = self.fc3(ch)
        return ch, h, x
        pass

    pass


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3)),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x):
        out = self.pool1(self.layer1(x))
        out = self.pool2(self.layer3(self.layer2(out)))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


batch_size = 64


class CifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, batch_size, 5)
        self.fc1 = nn.Linear(batch_size * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, batch_size * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


class EnhancedCifarNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
