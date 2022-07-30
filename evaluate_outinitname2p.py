import time
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.autograd as autograd
import data_loader_evaluate_outinitname
from torch.autograd import Variable
#from model import _G_xvz, _G_vzx
from model import _G_xvz, _G_vzx, _D_xvs, _G_vzx_withID
from itertools import *
import pdb
import ID_models.IdPreserving as ID_pre
import ID_models.MobileFaceNet as MBF

dd = pdb.set_trace

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_list", type=str, default="./list_test.txt")
parser.add_argument("-b", "--batch_size", type=int, default=2)
parser.add_argument('--outf', default='./evaluate', help='folder to output images and model checkpoints')
parser.add_argument('--modelf', default='./pretrained_model', help='folder to output images and model checkpoints')
parser.add_argument('--cuda', action='store_true', help='enables cuda', default=True)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

args = parser.parse_args()
print(args)

try:
    os.makedirs(args.outf)
except OSError:
    pass

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# need initialize!!
G_xvz = _G_xvz()
#G_vzx = _G_vzx()
G_vzx = _G_vzx_withID()
train_list = args.data_list

train_loader = torch.utils.data.DataLoader(
    data_loader_evaluate_outinitname.ImageList( train_list, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

def L1_loss(x, y):
    return torch.mean(torch.sum(torch.abs(x-y), 1))

# N*3*128*128(r,g,b) -> N*3*112*96(r,g,b)
def resize_img_to_MBF_input(args, img_tensor):
    loader = transforms.Compose([transforms.ToTensor()])
    unloader = transforms.ToPILImage()
    MBF_input_tensor = torch.FloatTensor(args.batch_size, 3, 112, 96)

    # convert tensor to PIL
    i = 0
    for img in img_tensor:
        PIL_img = img.cpu().clone()
        PIL_img = PIL_img.squeeze(0)
        PIL_img = unloader(PIL_img)
        # resize img
        PIL_img = PIL_img.resize((96, 112))
        # convert PIL to tensor
        image_tensor = loader(PIL_img).unsqueeze(0)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        image_tensor = image_tensor.to(device, torch.float)
        # combine
        MBF_input_tensor[i] = image_tensor
        i += 1

    return MBF_input_tensor

x = torch.FloatTensor(args.batch_size, 3, 128, 128)
x_bar_bar_out = torch.FloatTensor(2, 3, 128, 128)

v_siz = 9
z_siz = 128-v_siz
v = torch.FloatTensor(args.batch_size, v_siz)
z = torch.FloatTensor(args.batch_size, z_siz)

if args.cuda:
    G_xvz = torch.nn.DataParallel(G_xvz).cuda()
    G_vzx = torch.nn.DataParallel(G_vzx).cuda()

    x = x.cuda()
    x_bar_bar_out = x_bar_bar_out.cuda()
    v = v.cuda()
    z = z.cuda()

x = Variable(x)
x_bar_bar_out = Variable(x_bar_bar_out)
v = Variable(v)
z = Variable(z)

def load_pretrained_model(net, path, name):
    state_dict = torch.load('%s/%s' % (path,name))
    own_state = net.state_dict()

    for name, param in state_dict.items():
        if name not in own_state:
            print('not load weights %s' % name)
            continue
        own_state[name].copy_(param)

load_pretrained_model(G_xvz, args.modelf, 'netG_xvz.pth')
load_pretrained_model(G_vzx, args.modelf, 'netG_vzx.pth')

batch_size = args.batch_size
cudnn.benchmark = True
G_xvz.eval()
G_vzx.eval()
netR = ID_pre.define_R(gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7], \
    lightcnn_path='./ID_models/LightCNN_29Layers_V2_checkpoint.pth').cuda()
# initialize MobileFaceNet
MobileFacenet = MBF.MobileFacenet()
checkpoint = torch.load('/ssd01/wanghuijiao/F06All/ID_models/MobileFaceNet.ckpt')
MobileFacenet.load_state_dict(checkpoint['net_state_dict'])

x1 = torch.FloatTensor(args.batch_size, 3, 128, 128)
if args.cuda:
    x1 = x1.cuda()
x1 = Variable(x1)


for i, (data) in enumerate(train_loader):
    img = data[1]
    img_road = data[0][0]
    img_n = img_road.split('/')[-1]
    print(img_n)
    x.resize_(img.size()).copy_(img)

    x_bar_bar_out.data.zero_()
    v_bar, z_bar = G_xvz(x)
    '''
    for one_view in range(9):
        v.data.zero_()
        for d in range(data[1].size(0)):
            v.data[d][one_view] = 1
        exec('x_bar_bar_%d = G_vzx(v, z_bar, )' % (one_view))
    '''
    view_to_generate = 4
    v.data.zero_() 
    v.data[0][view_to_generate] = 1 
    x1.resize_(img.size()).copy_(img)
    img_ID_fea = netR(x1)
    mbf_x1 = resize_img_to_MBF_input(args, x1)
    mbf_ID_fea = MobileFacenet(mbf_x1)
    exec('x_bar_bar_%d = G_vzx(v, z_bar, img_ID_fea, mbf_ID_fea)' % (view_to_generate))
 
    for d in range(batch_size):
        x_bar_bar_out.data[0] = x.data[d]
        exec('x_bar_bar_out.data[1] = x_bar_bar_%d.data[d]' % (4))
        vutils.save_image(x_bar_bar_out.data,'%s/%s' % (args.outf, img_n), nrow = 2, normalize=True)
