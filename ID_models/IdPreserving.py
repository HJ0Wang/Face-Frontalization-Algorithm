import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import torchvision

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out


class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=80013, feature=True):
        super(network_29layers_v2, self).__init__()
        self.conv1 = mfm(1, 48, 5, 1, 2)
        self.block1 = self._make_layer(block, layers[0], 48, 48)
        self.group1 = group(48, 96, 3, 1, 1)
        self.block2 = self._make_layer(block, layers[1], 96, 96)
        self.group2 = group(96, 192, 3, 1, 1)
        self.block3 = self._make_layer(block, layers[2], 192, 192)
        self.group3 = group(192, 128, 3, 1, 1)
        self.block4 = self._make_layer(block, layers[3], 128, 128)
        self.group4 = group(128, 128, 3, 1, 1)
        self.fc = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)
        self.feature = feature

    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def makegray(self, input):
        x = input[:, 0, :, :] * 0.299 + input[:, 1, :, :] * 0.587 + input[:, 2, :, :] * 0.114  # to grayscale
        x = x.unsqueeze(1)
        return x

    def forward(self, input):
        # input: [B, 3 (r,g,b), 128, 128]
        x = self.makegray(input)

        # expected x: [4, 1, 128, 128]
        x = self.conv1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block1(x)
        x = self.group1(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block2(x)
        x = self.group2(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = self.block3(x)
        x = self.group3(x)
        x = self.block4(x)
        x = self.group4(x)
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2)

        x = x.view(x.size(0), -1)
        fc = self.fc(x)
        if self.feature:
            return fc
        x = F.dropout(fc, training=self.training)
        out = self.fc2(x)
        return out

def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model


# -----------------------------------------------------


def define_R(gpu_ids, lightcnn_path):
    # import models.modules.arch_lightcnn as arch
    # netR = arch.LightCNN_29Layers_v2()
    netR = LightCNN_29Layers_v2()
    netR.eval()
    if gpu_ids:
        netR = torch.nn.DataParallel(netR).cuda()
    checkpoint = torch.load(lightcnn_path)
    netR.load_state_dict(checkpoint['state_dict'])
    return netR


# ------------------------------------------------------
'''
netR = define_R(gpu_ids=[0, 1, 2, 3], lightcnn_path='/mnt/ficuszambia/jnli/facesr/recognition/pre-trained/LightCNN_29Layers_V2_checkpoint.pth').to(torch.device('cuda'))
cri_rec = nn.CosineEmbeddingLoss().to(torch.device('cuda'))

real_fea = netR(real).detach()
fake_fea = netR(fake)
l_g_rec = weight * cri_rec(fake_fea, real_fea, torch.ones(1).to(torch.device('cuda'))
l_g += l_g_rec
'''
