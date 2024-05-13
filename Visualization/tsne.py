from __future__ import print_function
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
import scipy.io
import os
from sklearn import manifold, datasets
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from utils import *
from PIL import Image
import torch.utils.data as data

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='regdb', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.03, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str, help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='llcm_agw_p4_n8_lr_0.1_seed_0_best.t', type=str, help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str, help='model save path')
parser.add_argument('--save_epoch', default=20, type=int, metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str, help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str, help='log save path')
parser.add_argument('--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int, metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int, metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int, metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int, metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str, metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--margin_cc', default=0.1, type=float, metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=10, type=int, help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int, metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int, metavar='t', help='random seed')
parser.add_argument('--dist-type', default='l2', type=str, help='type of distance')
parser.add_argument('--gpu', default='3', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../../data/SYSU-MM01/'
    n_class = 395
    test_mode = [1, 2]
elif dataset == 'regdb':
    data_path = '../../data/RegDB/'
    n_class = 206
    # test_mode = [2, 1]
    test_mode = [1, 2]

checkpoint_path = args.model_path

print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
best_map = 0  # best test map
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train_list = [
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]
transform_train = transforms.Compose(transform_train_list)
print(transform_train_list)


if not os.path.isdir('./save_tsne_regdb'):
    os.makedirs('./save_tsne_regdb')

result = scipy.io.loadmat('tsne.mat')
query_feature = torch.FloatTensor(result['query_f'])   # [3803, 2048]
query_label = result['query_label'][0]                 # [3803,]

gallery_feature = torch.FloatTensor(result['gallery_f']) # [301, 2048]
gallery_label = result['gallery_label'][0]  # [301,]
query_feature = query_feature.cuda()
gallery_feature = gallery_feature.cuda()
end = time.time()

class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        
        query_feature = torch.FloatTensor(result['query_f'])
        query_label = result['query_label'][0]
        
        gallery_feature = torch.FloatTensor(result['gallery_f'])
        gallery_label = result['gallery_label'][0]
        

        # BGR to RGB
        self.train_color_image = query_feature  
        self.train_color_label = query_label
        
        # BGR to RGB
        self.train_thermal_image = gallery_feature
        self.train_thermal_label = gallery_label
                
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)

# training set
trainset = SYSUData(data_path, transform=transform_train)
# generate the idx of each person identity
color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

tsne_class = len(np.unique(trainset.train_color_label))

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(tsne_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(tsne_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

def plot_embedding(X, y, z, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(15, 12), dpi=100)
    cx, cy = [], []

    r = []
    #print(X.shape[0])
    for i in range(X.shape[0]):
        # cx.append(X[i, 0])
        # cy.append(X[i, 1])
        if z[i] == 1:
            #print(i, y[i])
            plt.scatter(X[i, 0], X[i, 1], s=120, color=color[y[i]], edgecolor='black', marker='o')
        else:
            plt.scatter(X[i, 0], X[i, 1], s=120, color=color[y[i]], edgecolor='black', marker='^')
        plt.xticks([])
        plt.yticks([])

color= [(0.1, 0.1, 0.1, 1.0),#r, g, b
        (0.5, 0.5, 0.5, 1.0),
        (1.0, 0.6, 0.1, 1.0),
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 0.1, 0.7, 1.0),
        (0.9, 0.2, 0.4, 1.0),
        (0.8, 0.2, 1.0, 1.0),
        (0.8, 0.3, 0.2, 1.0),
        (0.7, 0.5, 0.3, 1.0),
        (0.7, 0.9, 0.4, 1.0),
        (0.7, 0.3, 0.8, 1.0),
        (0.0, 1.0, 0.0, 1.0),
        (0.4, 1.0, 0.6, 1.0),
        (0.3, 0.8, 0.5, 1.0),
        (0.1, 0.8, 1.0, 1.0),
        (0.5, 0.7, 0.9, 1.0),
        (0.4, 0.8, 0.3, 1.0),
        (0.5, 0.7, 0.4, 1.0),
        (0.2, 0.6, 0.8, 1.0),
        (0.1, 0.1, 1.0, 1.0),
        (0.3, 0.3, 0.9, 1.0),
        (0.6, 0.1, 0.4, 1.0),
        ]#R G B

        
def train(epoch):
    data_time = AverageMeter()
    # switch to train mode
    end = time.time()
    with torch.no_grad():
        for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
            labels = torch.cat((label1, label2), 0)
            z1 = torch.ones(label1.shape)
            z2 = torch.zeros(label2.shape)
            z = torch.cat((z1, z2), 0)
            print(batch_idx)
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())

            a = labels.unique()
            for i in range(len(a)):
                for j in range(len(labels)):
                    if labels[j] == a[i]:
                        labels[j] = i
            #print(labels)
            data_time.update(time.time() - end)
            out = torch.cat((input1, input2), 0)

            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            X_tsne = tsne.fit_transform(out.detach().cpu().numpy())
            plot_embedding(X_tsne, labels, z)
            plt.savefig(osp.join('save_tsne_regdb', 'tsne_GFMNet_{}.jpg'.format(batch_idx)))
          

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """
    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize, epoch):        
        # uni_label = np.unique(train_color_label)
        # self.n_classes = len(uni_label)
        uni_label = np.arange(0,96)

        N = np.maximum(len(train_color_label), len(train_thermal_label)) 
        for j in range(int(N/(batchSize*num_pos))+1):
            batch_idx = np.random.choice(uni_label, batchSize, replace = False)  
            for i in range(batchSize):
                sample_color  = np.random.choice(color_pos[batch_idx[i]], num_pos)
                sample_thermal = np.random.choice(thermal_pos[batch_idx[i]], num_pos)
                
                if j ==0 and i==0:
                    index1= sample_color
                    index2= sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))
        
        self.index1 = index1
        self.index2 = index2
        self.N  = N
        
    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N          


# training
print('==> Start Training...')
print('==> Preparing Data Loader...')
# identity sampler
epoch=0
print(trainset.train_color_label)
sampler = IdentitySampler(trainset.train_color_label, trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size, epoch)

trainset.cIndex = sampler.index1  # color index
trainset.tIndex = sampler.index2  # thermal index
print(epoch)
print(trainset.cIndex)
print(trainset.tIndex)

loader_batch = args.batch_size * args.num_pos

trainloader = data.DataLoader(trainset, batch_size=loader_batch, sampler=sampler, num_workers=args.workers, drop_last=True)
# training
train(epoch)
