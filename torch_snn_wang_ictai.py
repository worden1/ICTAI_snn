import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.backends import cudnn
from bisect import bisect_right
import math
import os


#命令行解析器部分，创建对象，add_argument，并parse_args()读入
parser = argparse.ArgumentParser(description='PyTorch local error training')
parser.add_argument('--model', default='vgg8b',
                    help='model, mlp, vgg13, vgg16, vgg19, vgg8b, vgg11b, resnet18, resnet34, wresnet28-10 and more (default: vgg8b)')
parser.add_argument('--dataset', default='CIFAR10',
                    help='dataset, MNIST, KuzushijiMNIST, FashionMNIST, CIFAR10, CIFAR100, SVHN, STL10 or ImageNet (default: CIFAR10)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--num-layers', type=int, default=1,
                    help='number of hidden fully-connected layers for mlp and vgg models (default: 1')
parser.add_argument('--num-hidden', type=int, default=1024,
                    help='number of hidden units for mpl model (default: 1024)')
parser.add_argument('--dim-in-decoder', type=int, default=4096,
                    help='input dimension of decoder_y used in pred and predsim loss (default: 4096)')
parser.add_argument('--feat-mult', type=float, default=1,
                    help='multiply number of CNN features with this number (default: 1)')
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs to train (default: 400)')
parser.add_argument('--classes-per-batch', type=int, default=0,
                    help='aim for this number of different classes per batch during training (default: 0, random batches)')
parser.add_argument('--classes-per-batch-until-epoch', type=int, default=0,
                    help='limit number of classes per batch until this epoch (default: 0, until end of training)')
parser.add_argument('--lr', type=float, default=5e-4,
                    help='initial learning rate (default: 5e-4)')
parser.add_argument('--lr-decay-milestones', nargs='+', type=int, default=[200,300,350,375],
                    help='decay learning rate at these milestone epochs (default: [200,300,350,375])')
parser.add_argument('--lr-decay-fact', type=float, default=0.25,
                    help='learning rate decay factor to use at milestone epochs (default: 0.25)')
parser.add_argument('--optim', default='adam',
                    help='optimizer, adam, amsgrad or sgd (default: adam)')
parser.add_argument('--momentum', type=float, default=0.0,
                    help='SGD momentum (default: 0.0)')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='weight decay (default: 0.0)')
parser.add_argument('--alpha', type=float, default=0.0,
                    help='unsupervised fraction in similarity matching loss (default: 0.0)')
parser.add_argument('--beta', type=float, default=0.99,
                    help='fraction of similarity matching loss in predsim loss (default: 0.99)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout after each nonlinearity (default: 0.0)')
parser.add_argument('--loss-sup', default='predsim',
                    help='supervised local loss, sim or pred (default: predsim)')
parser.add_argument('--loss-unsup', default='none',
                    help='unsupervised local loss, none, sim or recon (default: none)')
parser.add_argument('--nonlin', default='relu',
                    help='nonlinearity, relu or leakyrelu (default: relu)')
parser.add_argument('--no-similarity-std', action='store_true', default=False,
                    help='disable use of standard deviation in similarity matrix for feature maps')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disable CUDA training')
parser.add_argument('--backprop', action='store_true', default=False,
                    help='disable local loss training')
parser.add_argument('--no-batch-norm', action='store_true', default=False,
                    help='disable batch norm before non-linearities')
parser.add_argument('--no-detach', action='store_true', default=False,
                    help='do not detach computational graph')
parser.add_argument('--pre-act', action='store_true', default=False,
                    help='use pre-activation in ResNet')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
#parser.add_argument('--save-dir', default='/hdd/results/local-error', type=str,
#                    help='the directory used to save the trained models')
parser.add_argument('--save-dir', default='/home/wang/guoshiqi/data/results/local-error', type=str,
                    help='the directory used to save the trained models')
parser.add_argument('--resume', default='', type=str,
                    help='checkpoint to resume training from')
parser.add_argument('--progress-bar', action='store_true', default=False,
                    help='show progress bar during training')
parser.add_argument('--no-print-stats', action='store_true', default=False,
                    help='do not print layerwise statistics during training with local loss')
parser.add_argument('--bio', action='store_true', default=False,
                    help='use more biologically plausible versions of pred and sim loss (default: False)')
parser.add_argument('--target-proj-size', type=int, default=128,
                    help='size of target projection back to hidden layers for biologically plausible loss (default: 128')
parser.add_argument('--cutout', action='store_true', default=False,
                    help='apply cutout regularization')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the cutout holes in pixels')

args = parser.parse_args()
#cuda使用和随机种子使用部分
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    cudnn.enabled = True
    cudnn.benchmark = True
        
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

#随机mask裁剪图片
class Cutout(object):
    '''Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    '''
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    #系统自带的函数方法，用于使类的实例可以直接当作函数输入参数运行
    def __call__(self, img):
        '''
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        '''
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            #np.clip对数组按最小最大值区间裁剪
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        #torch.form_numpy将nparray转化为tensor
        mask = torch.from_numpy(mask)
        #将Tensor按照输入的tensor size扩张
        mask = mask.expand_as(img)
        img = img * mask

        return img
    
class NClassRandomSampler(torch.utils.data.sampler.Sampler):
    r'''Samples elements such that most batches have N classes per batch.
    Elements are shuffled before each epoch.

    Arguments:
        targets: target class for each example in the dataset
        n_classes_per_batch: the number of classes we want to have per batch
    '''

    def __init__(self, targets, n_classes_per_batch, batch_size):
        self.targets = targets
        self.n_classes = int(np.max(targets))
        self.n_classes_per_batch = n_classes_per_batch
        self.batch_size = batch_size

    #torch.utils.data.sampler.Sampler必须包含__iter__生成器迭代函数
    def __iter__(self):
        n = self.n_classes_per_batch
        
        ts = list(self.targets)
        ts_i = list(range(len(self.targets)))
        
        np.random.shuffle(ts_i)
        #algorithm outline: 
        #1) put n examples in batch
        #2) fill rest of batch with examples whose class is already in the batch
        while len(ts_i) > 0:
            idxs, ts_i = ts_i[:n], ts_i[n:] #pop n off the list
                
            t_slice_set = set([ts[i] for i in idxs])
            
            #fill up idxs until we have n different classes in it. this should be quick.
            k = 0
            while len(t_slice_set) < 10 and k < n*10 and k < len(ts_i):
                if ts[ts_i[k]] not in t_slice_set:
                    idxs.append(ts_i.pop(k))
                    t_slice_set = set([ts[i] for i in idxs])
                else:
                    k += 1
            
            #fill up idxs with indexes whose classes are in t_slice_set.
            j = 0
            while j < len(ts_i) and len(idxs) < self.batch_size:
                if ts[ts_i[j]] in t_slice_set:
                    idxs.append(ts_i.pop(j)) #pop is O(n), can we do better?
                else:
                    j += 1
            
            if len(idxs) < self.batch_size:
                needed = self.batch_size-len(idxs)
                idxs += ts_i[:needed]
                ts_i = ts_i[needed:]
                    
            for i in idxs:
                #yield可以简洁的得到一个不占大内存的生成迭代器generator
                yield i

    def __len__(self):
        return len(self.targets)
    
class KuzushijiMNIST(datasets.MNIST):
    urls = [
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/train-labels-idx1-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-images-idx3-ubyte.gz',
        'http://codh.rois.ac.jp/kmnist/dataset/kmnist/t10k-labels-idx1-ubyte.gz'
    ]

#可能是设置DataLoader中的进程数    
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
if args.dataset == 'MNIST':
    input_dim = 28
    input_ch = 1
    num_classes = 10
    #torchvision.transforms函数Compose:将多个transforms组合使用，随机中心切割，ToTensor转换为[C,H,W]0-1,以及给定均值方差的归一标准化
    train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length)) 
    dataset_train = datasets.MNIST('../data/MNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels.numpy(), args.classes_per_batch, args.batch_size),
        batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/MNIST', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'FashionMNIST':
    input_dim = 28
    input_ch = 1
    num_classes = 10
    train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.286,), (0.353,))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))      
    dataset_train = datasets.FashionMNIST('../data/FashionMNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels.numpy(), args.classes_per_batch, args.batch_size),
        batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data/FashionMNIST', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.286,), (0.353,))
            ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'KuzushijiMNIST':
    input_dim = 28
    input_ch = 1
    num_classes = 10
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1904,), (0.3475,))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    dataset_train = KuzushijiMNIST('../data/KuzushijiMNIST', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels.numpy(), args.classes_per_batch, args.batch_size),
        batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        KuzushijiMNIST('../data/KuzushijiMNIST', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1904,), (0.3475,))
            ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'CIFAR10':
    input_dim = 32
    input_ch = 1
    #input_ch = 3
    num_classes = 10
    train_transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            #transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
            transforms.Normalize((0.4809,), (0.1963,))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    dataset_train = datasets.CIFAR10('../data/CIFAR10', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels, args.classes_per_batch, args.batch_size),
        batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)

    '''
    mean = 0.
    std = 0.
    for images, _ in train_loader:
        batch_samples = images.size(0) # batch size (the last batch can have smaller size!)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= len(train_loader.dataset)
    std /= len(train_loader.dataset)
    print(mean,std)
    '''

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('../data/CIFAR10', train=False, 
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                #transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                transforms.Normalize((0.4809,), (0.1963,))
            ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    
    
elif args.dataset == 'CIFAR100':
    input_dim = 32
    input_ch = 3
    num_classes = 100
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    dataset_train = datasets.CIFAR100('../data/CIFAR100', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.train_labels, args.classes_per_batch, args.batch_size),
        batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100('../data/CIFAR100', train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.438, 0.418, 0.377), (0.300, 0.287, 0.294))
            ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)  
elif args.dataset == 'SVHN':
    input_dim = 32
    input_ch = 3
    num_classes = 10
    train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    dataset_train = torch.utils.data.ConcatDataset((
        datasets.SVHN('../data/SVHN', split='train', download=True, transform=train_transform),
        datasets.SVHN('../data/SVHN', split='extra', download=True, transform=train_transform)))
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.labels, args.classes_per_batch, args.batch_size),
        batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.SVHN('../data/SVHN', split='test', download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
            ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
elif args.dataset == 'STL10':
    input_dim = 96
    input_ch = 3
    num_classes = 10
    train_transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    dataset_train = datasets.STL10('../data/STL10', split='train', download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None if args.classes_per_batch == 0 else NClassRandomSampler(dataset_train.labels, args.classes_per_batch, args.batch_size),
        batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.STL10('../data/STL10', split='test', 
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.447, 0.440, 0.407), (0.260, 0.257, 0.271))
            ])),
        batch_size=args.batch_size, shuffle=False, **kwargs) 
elif args.dataset == 'ImageNet':
    input_dim = 224
    input_ch = 3
    num_classes = 1000
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    if args.cutout:
        train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length))
    dataset_train = datasets.ImageFolder('../data/ImageNet/train', transform=train_transform)
    labels = np.array([a[1] for a in dataset_train.samples])
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler = None if args.classes_per_batch == 0 else NClassRandomSampler(labels, args.classes_per_batch, args.batch_size),
        batch_size=args.batch_size, shuffle=args.classes_per_batch == 0, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('../data/ImageNet/val', 
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
else:
    print('No valid dataset is specified')

#继承扩展torch.autograd.Function类定义Function类，包括__init__,forward,backward方法，扩展Module包括多个Variable.
# 和Function形成计算图只有__init__，forward方法。其中必须有ctx做第一个输入用于保存等，之后forward输入输出也对应backward
# 的输出输入。    
class LinearFAFunction(torch.autograd.Function):
    '''Autograd function for linear feedback alignment module.
    '''
    @staticmethod
    def forward(context, input, weight, weight_fa, bias=None):
        context.save_for_backward(input, weight, weight_fa, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            #unsqueeze为在某维度上增加扩展维度
            output += bias.unsqueeze(0).expand_as(output)

        return output

    @staticmethod
    def backward(context, grad_output):
        input, weight, weight_fa, bias = context.saved_tensors
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if context.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight_fa)
        if context.needs_input_grad[1]:
            grad_weight = grad_output.t().matmul(input)
        if bias is not None and context.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias

#这里继承nn.Module扩展自定义线性模型；主要包括在定义好Function后的__init__和forward
class LinearFA(nn.Module):
    '''Linear feedback alignment module.

    Args:
        input_features (int): Number of input features to linear layer.
        output_features (int): Number of output features from linear layer.
        bias (bool): True if to use trainable bias.
    '''
    #def __init__(self, input_features, output_features, bias=True):
    def __init__(self, input_features, output_features, bias=False):
        super(LinearFA, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        self.weight_fa = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        #下边定义的reset_parameters函数    
        self.reset_parameters()
        
        if args.cuda:
            self.weight.data = self.weight.data.cuda()
            self.weight_fa.data = self.weight_fa.data.cuda()
            if bias:
                self.bias.data = self.bias.data.cuda()
    
    #初始化
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.weight_fa.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()
            
    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)
    
    #输出相关信息用于调试，self.__class__.__name__用于获得类的名字
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.input_features) \
            + ', out_features=' + str(self.output_features) \
            + ', bias=' + str(self.bias is not None) + ')'

snnlayer_config = {'num_bins': 4, 'dt': 3e-5, 'leak': 2, 'sigma_v': 0, 'mu': 0} 
class SnnFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, snnlayer_config):
        num_bins = snnlayer_config['num_bins']
        dt = snnlayer_config['dt']
        leak = snnlayer_config['leak']
        sigma_v = snnlayer_config['sigma_v']
        mu = snnlayer_config['mu']
        shape = input.shape

        #目前看input维度为(N,H,W,T)这里为(N,featuresmap)
        V_membrane = torch.zeros_like(input)
        #print(V_membrane.shape)
        #firing_rates = torch.zeros((shape[0], shape[1], num_bins))
        firing_rates = torch.zeros_like(input)
        #print(firing_rates.shape)
        #print(threshold)
        for t in range(num_bins - 1):
            # compute command signal
            #command_x = (images[:, :, t + 1] -
            #            images[:, :, t]) / dt + leak * images[:, :, t]
            #if t == 0:
            #    command_x = input / dt 
            #else:
            #    command_x = input * leak
            #考虑先改为输入不过差分有2snnbug的输入对应双层直接reset到0的方法可能和可解释方法都是对应有意义的

            # update membrane potential
            #V_membrane += dt * (-leak * V_membrane +
            #                    np.tensordot(F_weights, command_x, ([1,2],[0,1]))
            #                    ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)
            #V_membrane += dt * (-leak * V_membrane +
            #                    command_x
            #                    )
            V_membrane += dt * (-leak * V_membrane) + input

            # update rates
            firing_rates = (1 - leak * dt) * firing_rates
            #print('layer1--')
            flag = V_membrane > threshold
            flag = flag.float()

            V_membrane = V_membrane * (1 - flag)
            firing_rates += flag 

            ''' 
            # Check if any neurons are past their threshold during the last time-step
            diff_voltage_thresh = V_membrane - threshold
            spiking_neurons_indices = np.arange(N)[diff_voltage_thresh >= 0]

            if spiking_neurons_indices.size > 0:
                # Pick the neuron which likely would have spiked first, by max distance from threshold
                to_pick = np.argmax(V_membrane[spiking_neurons_indices] - threshold[spiking_neurons_indices])
                s = spiking_neurons_indices[to_pick]
                #for s in spiking_neurons_indices:
                # Update membrane potential
                V_membrane[s] -= 2 * (mu + V_membrane[s] - threshold[s])
                #V_membrane += omega[:, s]

                # Update rates with spikes
                firing_rates[s, t + 1] += 1
                #print('layer1spike')
            else:
                pass
            '''
            #V_mean = torch.from_numpy(V_membrane)
            #firing_rate = torch.mean(firing_rates, dim = 1)
        #firing_rateN = firing_rates.mm(torch.eye(N))
        ctx.save_for_backward(input, threshold, firing_rates)
        return firing_rates

    @staticmethod
    def backward(ctx, grad_output):
        input, threshold, firing_rates = ctx.saved_tensors
        #grad_input = grad_threshold = grad_snnlayer_config = None
        #if ctx.needs_input_grad[0]:
        grad_input = grad_output
        #if ctx.needs_input_grad[1]:
        #这里可能梯度方向有个反向的负号
        grad_threshold = -grad_output.t().mm(firing_rates)/1024
        #if ctx.needs_input_grad[2]:
            #grad_snnlayer_config = None
        return grad_input, grad_threshold, None

class SnnLayer(nn.Module):
    def __init__(self, input_features, output_features):
        super(SnnLayer, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.snnlayer_config = snnlayer_config
        self.threshold = nn.Parameter(torch.Tensor(input_features))

        self.reset_parameters()

        if args.cuda:
            self.threshold.data = self.threshold.data.cuda()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.threshold.size(0))
        self.threshold.data.uniform_(-stdv, stdv)       

    def forward(self, input):
        #忘记写apply发现很大bug调试了很久。。。
        return SnnFunction.apply(input, self.threshold, self.snnlayer_config)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.input_features) \
            + ', out_features=' + str(self.output_features) \
            + ')'


class SnnFunctionVf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, snnlayer_config):
        num_bins = snnlayer_config['num_bins']
        dt = snnlayer_config['dt']
        leak = snnlayer_config['leak']
        sigma_v = snnlayer_config['sigma_v']
        mu = snnlayer_config['mu']
        shape = input.shape

        #目前看input维度为(N,H,W,T)这里为(N,featuresmap)
        V_membrane = torch.zeros_like(input, requires_grad = False)
        #print(V_membrane.shape)
        #firing_rates = torch.zeros((shape[0], shape[1], num_bins))
        firing_rates = torch.zeros_like(input, requires_grad = False)
        #print(firing_rates.shape)
        #print(threshold.shape)
        for t in range(num_bins - 1):
            # compute command signal
            #command_x = (images[:, :, t + 1] -
            #            images[:, :, t]) / dt + leak * images[:, :, t]
            if t == 0:
                command_x = input / dt 
            else:
                #发现这里好像恰好脉冲一次后就稳态不变dv=0了所以增量改成2倍尝试一下可能之前改变不明显是通过+噪声实现的
                command_x = 2 * input * leak
            #考虑先改为输入不过差分有2snnbug的输入对应双层直接reset到0的方法可能和可解释方法都是对应有意义的

            # update membrane potential
            #V_membrane += dt * (-leak * V_membrane +
            #                    np.tensordot(F_weights, command_x, ([1,2],[0,1]))
            #                    ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)
            V_membrane += dt * (-leak * V_membrane +
                                command_x
                                )
            #V_membrane += dt * (-leak * V_membrane) + input

            # update rates
            #尝试删除leak firingrates，调试来看区别不大不过去掉后都是整数计数容易看些
            #firing_rates = (1 - leak * dt) * firing_rates
            #print('layer1--')
            #print(V_membrane.shape, threshold.shape)
            flag = V_membrane > threshold
            omega = ~ flag
            flag = flag.float()
            #omega = omega.float() * (1 - leak * dt)
            #这里可能抑制不能超过1倍否则又变成不能继续V上行然后脉冲了,print调试来看确实-1不动了不过会连续产生脉冲
            omega = omega.float() * (1 - leak * dt)
            

            #尝试reset到T-o(T)
            #V_membrane = V_membrane - 2 * (V_membrane - threshold) * flag
            #V_membrane = V_membrane - (V_membrane - threshold) * flag
            #尝试加入同城抑制的omega部分
            #V_membrane = (V_membrane - (V_membrane - threshold) * flag) * (flag + omega)
            #去掉同层抑制部分
            V_membrane = (V_membrane - (V_membrane - threshold) * flag)
            firing_rates += flag 
            #print(V_membrane, V_membrane.shape)
            ''' 
            # Check if any neurons are past their threshold during the last time-step
            diff_voltage_thresh = V_membrane - threshold
            spiking_neurons_indices = np.arange(N)[diff_voltage_thresh >= 0]

            if spiking_neurons_indices.size > 0:
                # Pick the neuron which likely would have spiked first, by max distance from threshold
                to_pick = np.argmax(V_membrane[spiking_neurons_indices] - threshold[spiking_neurons_indices])
                s = spiking_neurons_indices[to_pick]
                #for s in spiking_neurons_indices:
                # Update membrane potential
                V_membrane[s] -= 2 * (mu + V_membrane[s] - threshold[s])
                #V_membrane += omega[:, s]

                # Update rates with spikes
                firing_rates[s, t + 1] += 1
                #print('layer1spike')
            else:
                pass
            '''
            #V_mean = torch.from_numpy(V_membrane)
            #firing_rate = torch.mean(firing_rates, dim = 1)
        #firing_rateN = firing_rates.mm(torch.eye(N))
        firing_rates.detach_()
        #print(V_membrane, V_membrane.shape)
        #print(firing_rates, firing_rates.shape)
        ctx.save_for_backward(input, threshold, firing_rates)
        return V_membrane, firing_rates

    @staticmethod
    def backward(ctx, grad_output, grad_firingrates):
        input, threshold, firing_rates = ctx.saved_tensors
        #grad_input = grad_threshold = grad_snnlayer_config = None
        #if ctx.needs_input_grad[0]:
        #print(grad_output)
        #print(grad_output.shape)
        #grad_input = grad_output
        grad_input = grad_output
        #if ctx.needs_input_grad[1]:
        #这里可能梯度方向有个反向的负号
        #grad_threshold = - grad_output.t().mm(firing_rates)/128 + threshold * 3e-3
        #grad_threshold = grad_output.t().mm(firing_rates)
        #grad_threshold = (torch.sum(grad_output.mul(firing_rates), dim = 0) + threshold * 3e-3)
        #grad_threshold = - torch.sum(grad_output.mul(firing_rates), dim = 0) + 1e-4
        grad_threshold = - 1e7 * torch.mean(grad_output.mul(firing_rates), dim = 0) + 1e-1
        #grad_threshold = (1e3 * grad_output.mul(firing_rates) + threshold * 3e-3)
        #print(grad_threshold.shape)
        #if ctx.needs_input_grad[2]:
            #grad_snnlayer_config = None
        return grad_input, grad_threshold, None

class SnnLayerVf(nn.Module):
    def __init__(self, input_features, output_features):
        super(SnnLayerVf, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.snnlayer_config = snnlayer_config
        self.threshold = nn.Parameter(torch.Tensor(input_features))

        self.reset_parameters()

        if args.cuda:
            self.threshold.data = self.threshold.data.cuda()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.threshold.size(0))
        self.threshold.data.uniform_(-stdv, stdv)       

    def forward(self, input):
        #忘记写apply发现很大bug调试了很久。。。
        return SnnFunctionVf.apply(input, self.threshold, self.snnlayer_config)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.input_features) \
            + ', out_features=' + str(self.output_features) \
            + ')'


class SnnFunctionVfC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold, snnlayer_config):
        num_bins = snnlayer_config['num_bins']
        dt = snnlayer_config['dt']
        leak = snnlayer_config['leak']
        sigma_v = snnlayer_config['sigma_v']
        mu = snnlayer_config['mu']
        shape = input.shape

        #目前看input维度为(N,C,H,W,T)这里为(N,featuresmap)
        V_membrane = torch.zeros_like(input, requires_grad = False)
        #print(V_membrane.shape)
        #firing_rates = torch.zeros((shape[0], shape[1], num_bins))
        firing_rates = torch.zeros_like(input, requires_grad = False)
        #print(firing_rates.shape)
        #print(threshold.shape)
        for t in range(num_bins - 1):
            # compute command signal
            #command_x = (images[:, :, t + 1] -
            #            images[:, :, t]) / dt + leak * images[:, :, t]
            if t == 0:
                command_x = input / dt 
            else:
                #发现这里好像恰好脉冲一次后就稳态不变dv=0了所以增量改成2倍尝试一下可能之前改变不明显是通过+噪声实现的
                command_x = 2 * input * leak
            #考虑先改为输入不过差分有2snnbug的输入对应双层直接reset到0的方法可能和可解释方法都是对应有意义的

            # update membrane potential
            #V_membrane += dt * (-leak * V_membrane +
            #                    np.tensordot(F_weights, command_x, ([1,2],[0,1]))
            #                    ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)
            V_membrane += dt * (-leak * V_membrane +
                                command_x
                                )
            #V_membrane += dt * (-leak * V_membrane) + input

            # update rates
            #尝试删除leak firingrates，调试来看区别不大不过去掉后都是整数计数容易看些
            #firing_rates = (1 - leak * dt) * firing_rates
            #print('layer1--')
            #print(shape, V_membrane.shape, threshold.shape)
            temp = threshold.unsqueeze(-1).unsqueeze(-1).repeat(1, shape[2], shape[3])
            #print(temp, threshold)
            #flag = V_membrane > threshold.expand(shape[1], shape[2], shape[3])
            flag = V_membrane > temp
            omega = ~ flag
            flag = flag.float()
            #omega = omega.float() * (1 - leak * dt)
            #这里可能抑制不能超过1倍否则又变成不能继续V上行然后脉冲了,print调试来看确实-1不动了不过会连续产生脉冲
            omega = omega.float() * (1 - leak * dt)
            

            #尝试reset到T-o(T)
            #V_membrane = V_membrane - 2 * (V_membrane - threshold) * flag
            #V_membrane = V_membrane - (V_membrane - threshold) * flag
            #尝试加入同城抑制的omega部分
            #V_membrane = (V_membrane - (V_membrane - temp) * flag) * (flag + omega)
            #去掉同层抑制部分
            V_membrane = (V_membrane - (V_membrane - temp) * flag)
            firing_rates += flag 
            #print(V_membrane, V_membrane.shape)
            ''' 
            # Check if any neurons are past their threshold during the last time-step
            diff_voltage_thresh = V_membrane - threshold
            spiking_neurons_indices = np.arange(N)[diff_voltage_thresh >= 0]

            if spiking_neurons_indices.size > 0:
                # Pick the neuron which likely would have spiked first, by max distance from threshold
                to_pick = np.argmax(V_membrane[spiking_neurons_indices] - threshold[spiking_neurons_indices])
                s = spiking_neurons_indices[to_pick]
                #for s in spiking_neurons_indices:
                # Update membrane potential
                V_membrane[s] -= 2 * (mu + V_membrane[s] - threshold[s])
                #V_membrane += omega[:, s]

                # Update rates with spikes
                firing_rates[s, t + 1] += 1
                #print('layer1spike')
            else:
                pass
            '''
            #V_mean = torch.from_numpy(V_membrane)
            #firing_rate = torch.mean(firing_rates, dim = 1)
        #firing_rateN = firing_rates.mm(torch.eye(N))
        firing_rates.detach_()
        #print(V_membrane, V_membrane.shape)
        #print(firing_rates, firing_rates.shape)
        ctx.save_for_backward(input, threshold, firing_rates)
        return V_membrane, firing_rates

    @staticmethod
    def backward(ctx, grad_output, grad_firingrates):
        input, threshold, firing_rates = ctx.saved_tensors
        #grad_input = grad_threshold = grad_snnlayer_config = None
        #if ctx.needs_input_grad[0]:
        #print(grad_output)
        #print(grad_output.shape)
        #grad_input = grad_output
        grad_input = grad_output
        #if ctx.needs_input_grad[1]:
        #这里可能梯度方向有个反向的负号
        #grad_threshold = - grad_output.t().mm(firing_rates)/128 + threshold * 3e-3
        #grad_threshold = grad_output.t().mm(firing_rates)
        #grad_threshold = (torch.sum(grad_output.mul(firing_rates), dim = 0) + threshold * 3e-3)
        #grad_threshold = - torch.sum(grad_output.mul(firing_rates), dim = (0, 2, 3)) + 1e-4
        #grad_threshold = - 1e2 * torch.mean(grad_output.mul(firing_rates), dim = (0, 2, 3)) + 1e-4
        grad_threshold = - 1e7 * torch.mean(torch.sum(grad_output.mul(firing_rates), dim = (2, 3)), dim = 0) + 1e-1
        #grad_threshold = (1e3 * grad_output.mul(firing_rates) + threshold * 3e-3)
        #print(grad_threshold.shape)
        #if ctx.needs_input_grad[2]:
            #grad_snnlayer_config = None
        return grad_input, grad_threshold, None

class SnnLayerVfC(nn.Module):
    def __init__(self, ch_out, dim_out):
        super(SnnLayerVfC, self).__init__()
        self.ch_out = ch_out
        self.dim_out = dim_out
        self.snnlayer_config = snnlayer_config
        #可能这里之后要改为每一个ch同一的T
        self.threshold = nn.Parameter(torch.Tensor(ch_out))

        self.reset_parameters()

        if args.cuda:
            self.threshold.data = self.threshold.data.cuda()
    
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.threshold.size(0))
        self.threshold.data.uniform_(-stdv, stdv)       

    def forward(self, input):
        #忘记写apply发现很大bug调试了很久。。。
        return SnnFunctionVfC.apply(input, self.threshold, self.snnlayer_config)
    
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'ch_out=' + str(self.ch_out) \
            + ', dim_out=' + str(self.dim_out) \
            + ')'


class LocalLossBlockLinear(nn.Module):
    '''A module containing nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout
       The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
       
    Args:
        num_in (int): Number of input features to linear layer.
        num_out (int): Number of output features from linear layer.
        num_classes (int): Number of classes (used in local prediction loss).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        batchnorm (bool): True if to use batchnorm, if None, read from args.no_batch_norm.
    '''
    def __init__(self, num_in, num_out, num_classes, first_layer=False, dropout=None, batchnorm=None):
        super(LocalLossBlockLinear, self).__init__()
        
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        self.batchnorm = not args.no_batch_norm if batchnorm is None else batchnorm
        #self.encoder = nn.Linear(num_in, num_out, bias=True)
        self.encoder = nn.Linear(num_in, num_out, bias=False)
        
        if not args.backprop and args.loss_unsup == 'recon':
            #self.decoder_x = nn.Linear(num_out, num_in, bias=True)
            self.decoder_x = nn.Linear(num_out, num_in, bias=False)
        if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
            if args.bio:
                #可能这里修改后有所改进
                self.decoder_y = LinearFA(num_out, args.target_proj_size)
            else:
                #self.decoder_y = nn.Linear(num_out, num_classes)
                self.decoder_y = nn.Linear(num_out, num_classes, bias=False)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        if not args.backprop and not args.bio and (args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
            self.linear_loss = nn.Linear(num_out, num_out, bias=False)
        if self.batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_out)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif args.nonlin == 'snn':
            #self.nonlin = SnnLayer(num_out, num_out)
            self.nonlin = SnnLayerVf(num_out, num_out)
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
        if args.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam' or args.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.parameters(), lr=0, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
            
        self.clear_stats()
    
    def clear_stats(self):
        if not args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not args.backprop:
            stats = '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
                    self.encoder,
                    self.loss_sim / self.examples, 
                    self.loss_pred / self.examples,
                    100.0 * float(self.examples - self.correct) / self.examples,
                    self.examples)
            return stats
        else:
            return ''
    
            
    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def optim_zero_grad(self):
        self.optimizer.zero_grad()
        
    def optim_step(self):
        self.optimizer.step()
        
    def forward(self, x, y, y_onehot):        
        # The linear transformation
        h = self.encoder(x)
        
        # Add batchnorm and nonlinearity
        # 可能bn在SNN中要在nonlin之后
        if self.batchnorm:
            h = self.bn(h)
        #h = self.nonlin(h)
        if args.nonlin == 'snn':
            h, f = self.nonlin(h)
            h_return = f
        else:
            h = self.nonlin(h)
            h_return = h
        
        # Save return value and add dropout
        #h_return = f
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)
            
        # Calculate local loss and update weights
        #暂时看可能是这部分实现no bp在每次forward过程中就求好梯度下降了
        if (self.training or not args.no_print_stats) and not args.backprop:
            # Calculate hidden layer similarity matrix
            if args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim':
                if args.bio:
                    h_loss = h
                else:
                    h_loss = self.linear_loss(h)
                Rh = similarity_matrix(h_loss)
                
            # Calculate unsupervised loss
            if args.loss_unsup == 'sim':
                Rx = similarity_matrix(x).detach()
                loss_unsup = F.mse_loss(Rh, Rx)
            elif args.loss_unsup == 'recon' and not self.first_layer:
                x_hat = self.nonlin(self.decoder_x(h))
                loss_unsup = F.mse_loss(x_hat, x.detach())
            else:
                if args.cuda:
                    loss_unsup = torch.cuda.FloatTensor([0])
                else:
                    loss_unsup = torch.FloatTensor([0])
             
            # Calculate supervised loss
            if args.loss_sup == 'sim':
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                loss_sup = F.mse_loss(Rh, Ry)
                if not args.no_print_stats:
                    self.loss_sim += loss_sup.item() * h.size(0)
                    self.examples += h.size(0)
            elif args.loss_sup == 'pred':
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    float_type =  torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    loss_sup = F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                else:
                    loss_sup = F.cross_entropy(y_hat_local,  y.detach())
                if not args.no_print_stats:
                    self.loss_pred += loss_sup.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            elif args.loss_sup == 'predsim':                    
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                    float_type =  torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    loss_pred = (1-args.beta) * F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                    loss_pred = (1-args.beta) * F.cross_entropy(y_hat_local,  y.detach())
                loss_sim = args.beta * F.mse_loss(Rh, Ry)
                loss_sup = loss_pred + loss_sim
                if not args.no_print_stats:
                    self.loss_pred += loss_pred.item() * h.size(0)
                    self.loss_sim += loss_sim.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            
            # Combine unsupervised and supervised loss
            loss = args.alpha * loss_unsup + (1 - args.alpha) * loss_sup
                                             
            # Single-step back-propagation
            if self.training:
                loss.backward(retain_graph = args.no_detach)
            
            # Update weights in this layer and detatch computational graph
            if self.training and not args.no_detach:
                self.optimizer.step()
                self.optimizer.zero_grad()
                h_return.detach_()
                
            loss = loss.item()
        else:
            loss = 0.0
        
        return h_return, loss
    
class LocalLossBlockConv(nn.Module):
    '''
    A block containing nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d
    The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        ch_in (int): Number of input features maps.
        ch_out (int): Number of output features maps.
        kernel_size (int): Kernel size in Conv2d.
        stride (int): Stride in Conv2d.
        padding (int): Padding in Conv2d.
        num_classes (int): Number of classes (used in local prediction loss).
        dim_out (int): Feature map height/width for input (and output).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        bias (bool): True if to use trainable bias.
        pre_act (bool): True if to apply layer order nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d -> nn.Conv2d (used for PreActResNet).
        post_act (bool): True if to apply layer order nn.Conv2d -> nn.BatchNorm2d -> nn.ReLU -> nn.Dropou2d.
    '''
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding, num_classes, dim_out, first_layer=False, dropout=None, bias=None, pre_act=False, post_act=True):
        super(LocalLossBlockConv, self).__init__()
        
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        #self.bias = True if bias is None else bias
        self.bias = False
        self.pre_act = pre_act
        self.post_act = post_act
        self.encoder = nn.Conv2d(ch_in, ch_out, kernel_size, stride=stride, padding=padding, bias=self.bias)
            
        if not args.backprop and args.loss_unsup == 'recon':
            self.decoder_x = nn.ConvTranspose2d(ch_out, ch_in, kernel_size, stride=stride, padding=padding)
        if args.bio or (not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim')):
            # Resolve average-pooling kernel size in order for flattened dim to match args.dim_in_decoder
            ks_h, ks_w = 1, 1
            dim_out_h, dim_out_w = dim_out, dim_out
            dim_in_decoder = ch_out*dim_out_h*dim_out_w
            while dim_in_decoder > args.dim_in_decoder and ks_h < dim_out:
                ks_h*=2
                dim_out_h = math.ceil(dim_out / ks_h)
                dim_in_decoder = ch_out*dim_out_h*dim_out_w
                if dim_in_decoder > args.dim_in_decoder:
                   ks_w*=2
                   dim_out_w = math.ceil(dim_out / ks_w)
                   dim_in_decoder = ch_out*dim_out_h*dim_out_w 
            if ks_h > 1 or ks_w > 1:
                pad_h = (ks_h * (dim_out_h - dim_out // ks_h)) // 2
                pad_w = (ks_w * (dim_out_w - dim_out // ks_w)) // 2
                self.avg_pool = nn.AvgPool2d((ks_h,ks_w), padding=(pad_h, pad_w))
            else:
                self.avg_pool = None
        if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
            if args.bio:
                self.decoder_y = LinearFA(dim_in_decoder, args.target_proj_size)
            else:
                #self.decoder_y = nn.Linear(dim_in_decoder, num_classes)
                self.decoder_y = nn.Linear(dim_in_decoder, num_classes, bias=False)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        if not args.backprop and (args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
            #SNN may not need this
            self.conv_loss = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1, bias=False)
        if not args.no_batch_norm:
            if pre_act:
                self.bn_pre = torch.nn.BatchNorm2d(ch_in)
            if not (pre_act and args.backprop):
                self.bn = torch.nn.BatchNorm2d(ch_out)
                nn.init.constant_(self.bn.weight, 1)
                nn.init.constant_(self.bn.bias, 0)
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif args.nonlin == 'snn':
            self.nonlin = SnnLayerVfC(ch_out, dim_out)
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout2d(p=self.dropout_p, inplace=False)
        if args.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam' or args.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.parameters(), lr=0, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
        
        self.clear_stats()
    
    def clear_stats(self):
        if not args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not args.backprop:
            stats = '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
                    self.encoder,
                    self.loss_sim / self.examples, 
                    self.loss_pred / self.examples,
                    100.0 * float(self.examples - self.correct) / self.examples,
                    self.examples)
            return stats
        else:
            return ''
        
    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

    def optim_zero_grad(self):
        self.optimizer.zero_grad()
        
    def optim_step(self):
        self.optimizer.step()
        
    def forward(self, x, y, y_onehot, x_shortcut=None):
        # If pre-activation, apply batchnorm->nonlin->dropout
        if self.pre_act:
            if not args.no_batch_norm:
                x = self.bn_pre(x)
            x = self.nonlin(x)
            if self.dropout_p > 0:
                x = self.dropout(x)
            
        # The convolutional transformation
        h = self.encoder(x)
        
        # If post-activation, apply batchnorm
        if self.post_act and not args.no_batch_norm:
            h = self.bn(h)
            
        # Add shortcut branch (used in residual networks)
        if x_shortcut is not None:
            h = h + x_shortcut
            
        # If post-activation, add nonlinearity
        #if self.post_act:
        #    h = self.nonlin(h)
        if self.post_act:
            if args.nonlin == 'snn':
                h, f = self.nonlin(h)
                h_return = f
            else:
                h = self.nonlin(h)
                h_return = h
        else:
            h_return = h
        
        # Save return value and add dropout
        #h_return = h
        if self.post_act and self.dropout_p > 0:
            h_return = self.dropout(h_return)

        # Calculate local loss and update weights
        #暂时看可能是这部分实现no bp在每次forward过程中就求好梯度下降了
        if (not args.no_print_stats or self.training) and not args.backprop:
            # Add batchnorm and nonlinearity if not done already
            if not self.post_act:
                if not args.no_batch_norm:
                    h = self.bn(h)
                h = self.nonlin(h)

            # Calculate hidden feature similarity matrix
            if args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim':
                if args.bio:
                    h_loss = h
                    if self.avg_pool is not None:
                        h_loss = self.avg_pool(h_loss)
                else:
                    #h_loss = self.conv_loss(h)
                    h_loss = h
                Rh = similarity_matrix(h_loss)                    
          
            # Calculate unsupervised loss
            if args.loss_unsup == 'sim':
                Rx = similarity_matrix(x).detach()
                loss_unsup = F.mse_loss(Rh, Rx)
            elif args.loss_unsup == 'recon' and not self.first_layer:
                x_hat = self.nonlin(self.decoder_x(h))
                loss_unsup = F.mse_loss(x_hat, x.detach())
            else:
                if args.cuda:
                    loss_unsup = torch.cuda.FloatTensor([0])
                else:
                    loss_unsup = torch.FloatTensor([0])

            # Calculate supervised loss
            if args.loss_sup == 'sim':
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                loss_sup = F.mse_loss(Rh, Ry)
                if not args.no_print_stats:
                    self.loss_sim += loss_sup.item() * h.size(0)
                    self.examples += h.size(0)
            elif args.loss_sup == 'pred':
                if self.avg_pool is not None:
                    h = self.avg_pool(h)
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    float_type =  torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    loss_sup = F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                else:
                    loss_sup = F.cross_entropy(y_hat_local,  y.detach())
                if not args.no_print_stats:
                    self.loss_pred += loss_sup.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            elif args.loss_sup == 'predsim':
                if self.avg_pool is not None:
                    h = self.avg_pool(h)
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                    float_type =  torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    loss_pred = (1-args.beta) * F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                    loss_pred = (1-args.beta) * F.cross_entropy(y_hat_local,  y.detach())
                loss_sim = args.beta * F.mse_loss(Rh, Ry)
                loss_sup = loss_pred + loss_sim
                if not args.no_print_stats:
                    self.loss_pred += loss_pred.item() * h.size(0)
                    self.loss_sim += loss_sim.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
                
            # Combine unsupervised and supervised loss
            loss = args.alpha * loss_unsup + (1 - args.alpha) * loss_sup
                                             
            # Single-step back-propagation
            if self.training:
                loss.backward(retain_graph = args.no_detach)
                
            # Update weights in this layer and detatch computational graph
            if self.training and not args.no_detach:
                self.optimizer.step()
                self.optimizer.zero_grad()
                h_return.detach_()
                
            loss = loss.item()
        else:
            loss = 0.0
        
        return h_return, loss
    
class BasicBlock(nn.Module):
    ''' Used in ResNet() '''
    expansion = 1

    def __init__(self, in_planes, planes, stride, num_classes, input_dim):
        super(BasicBlock, self).__init__()
        self.input_dim = input_dim
        self.stride = stride
        self.conv1 = LocalLossBlockConv(in_planes, planes, 3, stride, 1, num_classes, input_dim, bias=False, pre_act=args.pre_act, post_act=not args.pre_act)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, 1, 1, num_classes, input_dim, bias=False, pre_act=args.pre_act, post_act=not args.pre_act)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False, groups=1),
                nn.BatchNorm2d(self.expansion*planes)
            )
            if args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')

    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        if len(self.shortcut) > 0:
              for param_group in self.optimizer.param_groups:
                  param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        if len(self.shortcut) > 0:
              self.optimizer.zero_grad()
        
    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        if len(self.shortcut) > 0:
              self.optimizer.step()
        
    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out,loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        out,loss = self.conv2(out, y, y_onehot, self.shortcut(x)) 
        loss_total += loss
        if not args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
              
        return (out, y, y_onehot, loss_total)

class Bottleneck(nn.Module):
    ''' Used in ResNet() '''
    expansion = 4

    def __init__(self, in_planes, planes, stride, num_classes, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = LocalLossBlockConv(in_planes, planes, 1, 1, 0, num_classes, input_dim, bias=False)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, stride, 1, num_classes, input_dim//stride, bias=False)
        self.conv3 = LocalLossBlockConv(planes, self.expansion*planes, 1, 1, 0, num_classes, input_dim//stride, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
            if args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
    
    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        self.conv3.set_learning_rate(lr)
        if len(self.shortcut) > 0:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        self.conv3.optim_zero_grad()
        if len(self.shortcut) > 0:
              self.optimizer.zero_grad()
        
    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        self.conv3.optim_step()
        if len(self.shortcut) > 0:
              self.optimizer.step()
              
    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out,loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        out, loss = self.conv2(out, y, y_onehot)  
        loss_total += loss
        out, loss = self.conv3(out, y, y_onehot, self.shortcut(x))
        loss_total += loss
         
        if not args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        return (out, y, y_onehot, loss_total)

class ResNet(nn.Module):
    '''
    Residual network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    '''
    def __init__(self, block, num_blocks, num_classes, input_ch, feature_multiplyer, input_dim):
        super(ResNet, self).__init__()
        self.in_planes = int(feature_multiplyer*64)
        self.conv1 = LocalLossBlockConv(input_ch, int(feature_multiplyer*64), 3, 1, 1, num_classes, input_dim, bias=False, post_act=not args.pre_act)
        self.layer1 = self._make_layer(block, int(feature_multiplyer*64), num_blocks[0], 1, num_classes, input_dim)
        self.layer2 = self._make_layer(block, int(feature_multiplyer*128), num_blocks[1], 2, num_classes, input_dim)
        self.layer3 = self._make_layer(block, int(feature_multiplyer*256), num_blocks[2], 2, num_classes, input_dim//2)
        self.layer4 = self._make_layer(block, int(feature_multiplyer*512), num_blocks[3], 2, num_classes, input_dim//4)
        self.linear = nn.Linear(int(feature_multiplyer*512*block.expansion), num_classes)
        if not args.backprop:
            self.linear.weight.data.zero_()
            
    def _make_layer(self, block, planes, num_blocks, stride, num_classes, input_dim):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        stride_cum = 1
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, num_classes, input_dim//stride_cum))
            stride_cum *= stride
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def parameters(self):
        if not args.backprop:
            return self.linear.parameters()
        else:
            return super(ResNet, self).parameters()
    
    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        for layer in self.layer1:
            layer.set_learning_rate(lr)
        for layer in self.layer2:
            layer.set_learning_rate(lr)
        for layer in self.layer3:
            layer.set_learning_rate(lr)
        for layer in self.layer4:
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        for layer in self.layer1:
            layer.optim_zero_grad()
        for layer in self.layer2:
            layer.optim_zero_grad()
        for layer in self.layer3:
            layer.optim_zero_grad()
        for layer in self.layer4:
            layer.optim_zero_grad()
        
    def optim_step(self):
        self.conv1.optim_step()
        for layer in self.layer1:
            layer.optim_step()
        for layer in self.layer2:
            layer.optim_step()
        for layer in self.layer3:
            layer.optim_step()
        for layer in self.layer4:
            layer.optim_step()
              
    def forward(self, x, y, y_onehot):      
        x,loss = self.conv1(x, y, y_onehot)
        x,_,_,loss = self.layer1((x, y, y_onehot, loss))
        x,_,_,loss = self.layer2((x, y, y_onehot, loss))
        x,_,_,loss = self.layer3((x, y, y_onehot, loss))
        x,_,_,loss = self.layer4((x, y, y_onehot, loss))
        x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x, loss

class wide_basic(nn.Module):
    ''' Used in WideResNet() '''
    def __init__(self, in_planes, planes, dropout_rate, stride, num_classes, input_dim, adapted):
        super(wide_basic, self).__init__()
        self.adapted = adapted
        self.conv1 = LocalLossBlockConv(in_planes, planes, 3, 1, 1, num_classes, input_dim*stride, dropout=None if self.adapted else 0, bias=True, pre_act=True, post_act=False)
        if not self.adapted:
            self.dropout = nn.Dropout(p=dropout_rate)
        self.conv2 = LocalLossBlockConv(planes, planes, 3, stride, 1, num_classes, input_dim, dropout=None if self.adapted else 0, bias=True, pre_act=True, post_act=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )
            if args.optim == 'sgd':
                self.optimizer = optim.SGD(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
            elif args.optim == 'adam' or args.optim == 'amsgrad':
                self.optimizer = optim.Adam(self.shortcut.parameters(), lr=0, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')

    def set_learning_rate(self, lr):
        self.lr = lr
        self.conv1.set_learning_rate(lr)
        self.conv2.set_learning_rate(lr)
        if len(self.shortcut) > 0:
              for param_group in self.optimizer.param_groups:
                  param_group['lr'] = lr

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        self.conv2.optim_zero_grad()
        if len(self.shortcut) > 0:
              self.optimizer.zero_grad()

    def optim_step(self):
        self.conv1.optim_step()
        self.conv2.optim_step()
        if len(self.shortcut) > 0:
              self.optimizer.step()

    def forward(self, input):
        x, y, y_onehot, loss_total = input
        out,loss = self.conv1(x, y, y_onehot)
        loss_total += loss
        if not self.adapted:
            out = self.dropout(out)
        out,loss = self.conv2(out, y, y_onehot, self.shortcut(x))
        loss_total += loss
        if not args.no_detach:
            if len(self.shortcut) > 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
     
        return (out, y, y_onehot, loss_total)

class Wide_ResNet(nn.Module):

    '''
    Wide residual network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    '''
    def __init__(self, depth, widen_factor, dropout_rate, num_classes, input_ch, input_dim, adapted=False):
        super(Wide_ResNet, self).__init__()
        self.adapted = adapted
        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = int((depth-4)/6)
        k = widen_factor

        print('| Wide-Resnet %dx%d %s' %(depth, k, 'adapted' if adapted else ''))
        if self.adapted:
            nStages = [16*k, 16*k, 32*k, 64*k]
        else:
            nStages = [16, 16*k, 32*k, 64*k]
        self.in_planes = nStages[0]

        self.conv1 = LocalLossBlockConv(input_ch, nStages[0], 3, 1, 1, num_classes, 32, dropout=0, bias=True, post_act=False) 
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, 1, num_classes, input_dim, adapted)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, 2, num_classes, input_dim, adapted)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, 2, num_classes, input_dim//2, adapted)
        self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
        self.linear = nn.Linear(nStages[3]*(16 if self.adapted else 1), num_classes)
        if not args.backprop:
            self.linear.weight.data.zero_()
            
    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, num_classes, input_dim, adapted):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        stride_cum = 1
        for stride in strides:
            stride_cum *= stride
            layers.append(block(self.in_planes, planes, dropout_rate, stride, num_classes, input_dim//stride_cum, adapted))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def parameters(self):
        if not args.backprop:
            return self.linear.parameters()
        else:
            return super(Wide_ResNet, self).parameters()
    
    def set_learning_rate(self, lr):
        self.conv1.set_learning_rate(lr)
        for layer in self.layer1:
            layer.set_learning_rate(lr)
        for layer in self.layer2:
            layer.set_learning_rate(lr)
        for layer in self.layer3:
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        self.conv1.optim_zero_grad()
        for layer in self.layer1:
            layer.optim_zero_grad()
        for layer in self.layer2:
            layer.optim_zero_grad()
        for layer in self.layer3:
            layer.optim_zero_grad()
        
    def optim_step(self):
        self.conv1.optim_step()
        for layer in self.layer1:
            layer.optim_step()
        for layer in self.layer2:
            layer.optim_step()
        for layer in self.layer3:
            layer.optim_step()
            
    def forward(self, x, y, y_onehot):
        x,loss = self.conv1(x, y, y_onehot)
        x,_,_,loss = self.layer1((x, y, y_onehot, loss))
        x,_,_,loss = self.layer2((x, y, y_onehot, loss))
        x,_,_,loss = self.layer3((x, y, y_onehot, loss))
        x = F.relu(self.bn1(x))
        if self.adapted:
            x = F.max_pool2d(x, 2)
        else:
            x = F.avg_pool2d(x, 8)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x, loss

#这里为使用线性层尝试构建的全连接model，no bp下每层前向时分别求梯度，之前的和之后的分别为conv变种和线性变种，整体大概看完了整个实现结构。
class Net(nn.Module):
    '''
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    '''
    def __init__(self, num_layers, num_hidden, input_dim, input_ch, num_classes):
        super(Net, self).__init__()
        
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList([LocalLossBlockLinear(input_dim*input_dim*input_ch, num_hidden, num_classes, first_layer=True)])
        self.layers.extend([LocalLossBlockLinear(int(num_hidden // (reduce_factor**(i-1))), int(num_hidden // (reduce_factor**i)), num_classes) for i in range(1, num_layers)])
        self.layer_out = nn.Linear(int(num_hidden // (reduce_factor**(num_layers-1))), num_classes)
        #self.layer_out = nn.Linear(int(num_hidden // (reduce_factor**(num_layers-1))), num_classes, bias = False)
        if not args.backprop:
            self.layer_out.weight.data.zero_()
            
    def parameters(self):
        if not args.backprop:
            return self.layer_out.parameters()
        else:
            return super(Net, self).parameters()
    
    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.optim_zero_grad()
        
    def optim_step(self):
        for i, layer in enumerate(self.layers):
            layer.optim_step()
            
    def forward(self, x, y, y_onehot):
        x = x.view(x.size(0), -1)
        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, loss = layer(x, y, y_onehot)
            total_loss += loss
        x = self.layer_out(x)

        return x, total_loss


class LocalLossBlockLinearNosnn(nn.Module):
    '''A module containing nn.Linear -> nn.BatchNorm1d -> nn.ReLU -> nn.Dropout
       The block can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
       
    Args:
        num_in (int): Number of input features to linear layer.
        num_out (int): Number of output features from linear layer.
        num_classes (int): Number of classes (used in local prediction loss).
        first_layer (bool): True if this is the first layer in the network (used in local reconstruction loss).
        dropout (float): Dropout rate, if None, read from args.dropout.
        batchnorm (bool): True if to use batchnorm, if None, read from args.no_batch_norm.
    '''
    def __init__(self, num_in, num_out, num_classes, first_layer=False, dropout=None, batchnorm=None):
        super(LocalLossBlockLinearNosnn, self).__init__()
        
        self.num_classes = num_classes
        self.first_layer = first_layer
        self.dropout_p = args.dropout if dropout is None else dropout
        self.batchnorm = not args.no_batch_norm if batchnorm is None else batchnorm
        self.encoder = nn.Linear(num_in, num_out, bias=True)
        #self.encoder = nn.Linear(num_in, num_out, bias=False)
        
        if not args.backprop and args.loss_unsup == 'recon':
            self.decoder_x = nn.Linear(num_out, num_in, bias=True)
            #self.decoder_x = nn.Linear(num_out, num_in, bias=False)
        if not args.backprop and (args.loss_sup == 'pred' or args.loss_sup == 'predsim'):
            if args.bio:
                #可能这里修改后有所改进
                self.decoder_y = LinearFA(num_out, args.target_proj_size)
            else:
                self.decoder_y = nn.Linear(num_out, num_classes)
                #self.decoder_y = nn.Linear(num_out, num_classes, bias=False)
            self.decoder_y.weight.data.zero_()
        if not args.backprop and args.bio:
            self.proj_y = nn.Linear(num_classes, args.target_proj_size, bias=False)
        if not args.backprop and not args.bio and (args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim'):
            self.linear_loss = nn.Linear(num_out, num_out, bias=False)
        if self.batchnorm:
            self.bn = torch.nn.BatchNorm1d(num_out)
            nn.init.constant_(self.bn.weight, 1)
            nn.init.constant_(self.bn.bias, 0)
        if args.nonlin == 'relu':
            self.nonlin = nn.ReLU(inplace=True)
        elif args.nonlin == 'leakyrelu':
            self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif args.nonlin == 'snn':
            #self.nonlin = SnnLayer(num_out, num_out)
            #self.nonlin = SnnLayerVf(num_out, num_out)
            self.nonlin = nn.ReLU(inplace=True)
        if self.dropout_p > 0:
            self.dropout = torch.nn.Dropout(p=self.dropout_p, inplace=False)
        if args.optim == 'sgd':
            self.optimizer = optim.SGD(self.parameters(), lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
        elif args.optim == 'adam' or args.optim == 'amsgrad':
            self.optimizer = optim.Adam(self.parameters(), lr=0, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
            
        self.clear_stats()
    
    def clear_stats(self):
        if not args.no_print_stats:
            self.loss_sim = 0.0
            self.loss_pred = 0.0
            self.correct = 0
            self.examples = 0

    def print_stats(self):
        if not args.backprop:
            stats = '{}, loss_sim={:.4f}, loss_pred={:.4f}, error={:.3f}%, num_examples={}\n'.format(
                    self.encoder,
                    self.loss_sim / self.examples, 
                    self.loss_pred / self.examples,
                    100.0 * float(self.examples - self.correct) / self.examples,
                    self.examples)
            return stats
        else:
            return ''
    
            
    def set_learning_rate(self, lr):
        self.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
    
    def optim_zero_grad(self):
        self.optimizer.zero_grad()
        
    def optim_step(self):
        self.optimizer.step()
        
    def forward(self, x, y, y_onehot):        
        # The linear transformation
        h = self.encoder(x)
        
        # Add batchnorm and nonlinearity
        # 可能bn在SNN中要在nonlin之后
        if self.batchnorm:
            h = self.bn(h)
        h = self.nonlin(h)
        h_return = h
        #if args.nonlin == 'snn':
        #    h, f = self.nonlin(h)
        #    h_return = f
        #else:
        #    h = self.nonlin(h)
        #    h_return = h
        
        # Save return value and add dropout
        #h_return = f
        if self.dropout_p > 0:
            h_return = self.dropout(h_return)
            
        # Calculate local loss and update weights
        #暂时看可能是这部分实现no bp在每次forward过程中就求好梯度下降了
        if (self.training or not args.no_print_stats) and not args.backprop:
            # Calculate hidden layer similarity matrix
            if args.loss_unsup == 'sim' or args.loss_sup == 'sim' or args.loss_sup == 'predsim':
                if args.bio:
                    h_loss = h
                else:
                    h_loss = self.linear_loss(h)
                Rh = similarity_matrix(h_loss)
                
            # Calculate unsupervised loss
            if args.loss_unsup == 'sim':
                Rx = similarity_matrix(x).detach()
                loss_unsup = F.mse_loss(Rh, Rx)
            elif args.loss_unsup == 'recon' and not self.first_layer:
                x_hat = self.nonlin(self.decoder_x(h))
                loss_unsup = F.mse_loss(x_hat, x.detach())
            else:
                if args.cuda:
                    loss_unsup = torch.cuda.FloatTensor([0])
                else:
                    loss_unsup = torch.FloatTensor([0])
             
            # Calculate supervised loss
            if args.loss_sup == 'sim':
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                loss_sup = F.mse_loss(Rh, Ry)
                if not args.no_print_stats:
                    self.loss_sim += loss_sup.item() * h.size(0)
                    self.examples += h.size(0)
            elif args.loss_sup == 'pred':
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    float_type =  torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    loss_sup = F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                else:
                    loss_sup = F.cross_entropy(y_hat_local,  y.detach())
                if not args.no_print_stats:
                    self.loss_pred += loss_sup.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            elif args.loss_sup == 'predsim':                    
                y_hat_local = self.decoder_y(h.view(h.size(0), -1))
                if args.bio:
                    Ry = similarity_matrix(self.proj_y(y_onehot)).detach()
                    float_type =  torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
                    y_onehot_pred = self.proj_y(y_onehot).gt(0).type(float_type).detach()
                    loss_pred = (1-args.beta) * F.binary_cross_entropy_with_logits(y_hat_local, y_onehot_pred)
                else:
                    Ry = similarity_matrix(y_onehot).detach()
                    loss_pred = (1-args.beta) * F.cross_entropy(y_hat_local,  y.detach())
                loss_sim = args.beta * F.mse_loss(Rh, Ry)
                loss_sup = loss_pred + loss_sim
                if not args.no_print_stats:
                    self.loss_pred += loss_pred.item() * h.size(0)
                    self.loss_sim += loss_sim.item() * h.size(0)
                    self.correct += y_hat_local.max(1)[1].eq(y).cpu().sum()
                    self.examples += h.size(0)
            
            # Combine unsupervised and supervised loss
            loss = args.alpha * loss_unsup + (1 - args.alpha) * loss_sup
                                             
            # Single-step back-propagation
            if self.training:
                loss.backward(retain_graph = args.no_detach)
            
            # Update weights in this layer and detatch computational graph
            if self.training and not args.no_detach:
                self.optimizer.step()
                self.optimizer.zero_grad()
                h_return.detach_()
                
            loss = loss.item()
        else:
            loss = 0.0
        
        return h_return, loss


class NetL(nn.Module):
    '''
    A fully connected network.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        num_layers (int): Number of hidden layers.
        num_hidden (int): Number of units in each hidden layer.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
    '''
    def __init__(self, num_layers, num_hidden, input_dim, input_ch, num_classes):
        super(NetL, self).__init__()
        
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        reduce_factor = 1
        self.layers = nn.ModuleList([LocalLossBlockLinearNosnn(input_dim*input_dim*input_ch, num_hidden, num_classes, first_layer=True)])
        self.layers.extend([LocalLossBlockLinearNosnn(int(num_hidden // (reduce_factor**(i-1))), int(num_hidden // (reduce_factor**i)), num_classes) for i in range(1, num_layers)])
        self.layer_out = nn.Linear(int(num_hidden // (reduce_factor**(num_layers-1))), num_classes)
        #self.layer_out = nn.Linear(int(num_hidden // (reduce_factor**(num_layers-1))), num_classes, bias = False)
        if not args.backprop:
            self.layer_out.weight.data.zero_()
            
    def parameters(self):
        if not args.backprop:
            return self.layer_out.parameters()
        else:
            return super(NetL, self).parameters()
    
    def set_learning_rate(self, lr):
        for i, layer in enumerate(self.layers):
            layer.set_learning_rate(lr)

    def optim_zero_grad(self):
        for i, layer in enumerate(self.layers):
            layer.optim_zero_grad()
        
    def optim_step(self):
        for i, layer in enumerate(self.layers):
            layer.optim_step()
            
    def forward(self, x, y, y_onehot):
        x = x.view(x.size(0), -1)
        total_loss = 0.0
        for i, layer in enumerate(self.layers):
            x, loss = layer(x, y, y_onehot)
            total_loss += loss
        x = self.layer_out(x)

        return x, total_loss


cfg = {
    'vgg6a':  [128, 'M', 256, 'M', 512, 'M', 512],
    'vgg6b':  [128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8':   [ 64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 'M'],
    'vgg8a':  [128, 256, 'M', 256, 512, 'M', 512, 'M', 512],
    'vgg8b':  [128, 256, 'M', 256, 512, 'M', 512, 'M', 512, 'M'],
    'vgg11':  [ 64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg11a': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512],
    'vgg11b': [128, 128, 128, 256, 'M', 256, 512, 'M', 512, 512, 'M', 512, 'M'],
    'vgg13':  [ 64,  64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16':  [ 64,  64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19':  [ 64,  64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGGn(nn.Module):
    '''
    VGG and VGG-like networks.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        vgg_name (str): The name of the network.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
        feat_mult (float): Multiply number of feature maps with this number.
    '''
    def __init__(self, vgg_name, input_dim, input_ch, num_classes, feat_mult=1):
        super(VGGn, self).__init__()
        self.cfg = cfg[vgg_name]
        self.input_dim = input_dim
        self.input_ch = input_ch
        self.num_classes = num_classes
        self.features, output_dim = self._make_layers(self.cfg, input_ch, input_dim, feat_mult)
        for layer in self.cfg:
            if isinstance(layer, int):
                output_ch = layer
        if args.num_layers > 0: 
            self.classifier = NetL(args.num_layers, args.num_hidden, output_dim, int(output_ch * feat_mult), num_classes)
        else:
            self.classifier = nn.Linear(output_dim*output_dim*int(output_ch * feat_mult), num_classes)
            
    def parameters(self):
        if not args.backprop:
            return self.classifier.parameters()
        else:
            return super(VGGn, self).parameters()

    def set_learning_rate(self, lr):
        for i,layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].set_learning_rate(lr)
        if args.num_layers > 0:
            self.classifier.set_learning_rate(lr)
            
    def optim_zero_grad(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_zero_grad()
        if args.num_layers > 0:
            self.classifier.optim_zero_grad()
            
    def optim_step(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_step()
        if args.num_layers > 0:
            self.classifier.optim_step()
            
    def forward(self, x, y, y_onehot):
        loss_total = 0
        for i,layer in enumerate(self.cfg):
            if isinstance(layer, int):
                x, loss = self.features[i](x, y, y_onehot)
                loss_total += loss  
            else:
                x = self.features[i](x)
        
        if args.num_layers > 0:         
            x, loss = self.classifier(x, y, y_onehot)
        else:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        loss_total += loss
        
        return x, loss_total

    def _make_layers(self, cfg, input_ch, input_dim, feat_mult):
        layers = []
        first_layer = True
        scale_cum = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                scale_cum *=2
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *=2
            elif x == 'M4':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
                scale_cum *=4
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                scale_cum *=2
            elif x == 'A3':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *=2
            elif x == 'A4':
                layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
                scale_cum *=4
            else:
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    scale_cum = 2
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=7, stride=2, padding=3, 
                                             num_classes=num_classes, 
                                             dim_out=input_dim//scale_cum, 
                                             first_layer=first_layer)]
                else:
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=3, stride=1, padding=1, 
                                             num_classes=num_classes, 
                                             dim_out=input_dim//scale_cum, 
                                             first_layer=first_layer)]
                input_ch = x
                first_layer = False
        
        return nn.Sequential(*layers), input_dim//scale_cum


cfg1 = {
    'snn1':  [15, 'M', 40, 'M', 300],
    'snn1m':  [15, 'M', 40, 'M'],
    'snn2':  [32, 'M', 64, 'M', 1024],
    'snn2m':  [32, 'M', 64, 'M'],
    'snn3m':  [96, 256, 'M', 384, 'M', 384, 256],
    'snn4m':  [128, 256, 'M', 512, 'M', 1024, 512],
}

class SNNn(nn.Module):
    '''
    VGG and VGG-like networks.
    The network can be trained by backprop or by locally generated error signal based on cross-entropy and/or similarity matching loss.
    
    Args:
        vgg_name (str): The name of the network.
        input_dim (int): Feature map height/width for input.
        input_ch (int): Number of feature maps for input.
        num_classes (int): Number of classes (used in local prediction loss).
        feat_mult (float): Multiply number of feature maps with this number.
    '''
    def __init__(self, vgg_name, input_dim, input_ch, num_classes, feat_mult=1):
        super(SNNn, self).__init__()
        self.cfg = cfg1[vgg_name]
        self.input_dim = input_dim
        self.input_ch = input_ch
        self.num_classes = num_classes
        self.features, output_dim = self._make_layers(self.cfg, input_ch, input_dim, feat_mult)
        for layer in self.cfg:
            if isinstance(layer, int):
                output_ch = layer
        if args.num_layers > 0: 
            self.classifier = NetL(args.num_layers, args.num_hidden, output_dim, int(output_ch * feat_mult), num_classes)
        else:
            self.classifier = nn.Linear(output_dim*output_dim*int(output_ch * feat_mult), num_classes)
            
    def parameters(self):
        if not args.backprop:
            return self.classifier.parameters()
        else:
            return super(SNNn, self).parameters()

    def set_learning_rate(self, lr):
        for i,layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].set_learning_rate(lr)
        if args.num_layers > 0:
            self.classifier.set_learning_rate(lr)
            
    def optim_zero_grad(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_zero_grad()
        if args.num_layers > 0:
            self.classifier.optim_zero_grad()
            
    def optim_step(self):
        for i, layer in enumerate(self.cfg):
            if isinstance(layer, int):
                self.features[i].optim_step()
        if args.num_layers > 0:
            self.classifier.optim_step()
            
    def forward(self, x, y, y_onehot):
        loss_total = 0
        for i,layer in enumerate(self.cfg):
            if isinstance(layer, int):
                x, loss = self.features[i](x, y, y_onehot)
                loss_total += loss  
            else:
                x = self.features[i](x)
        
        if args.num_layers > 0:         
            x, loss = self.classifier(x, y, y_onehot)
        else:
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
        loss_total += loss
        
        return x, loss_total

    def _make_layers(self, cfg, input_ch, input_dim, feat_mult):
        layers = []
        first_layer = True
        scale_cum = 1
        dimo_temp = input_dim
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                scale_cum *=2
                dimo_temp = dimo_temp//2
            elif x == 'M3':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *=2
            elif x == 'M4':
                layers += [nn.MaxPool2d(kernel_size=4, stride=4)]
                scale_cum *=4
            elif x == 'A':
                layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                scale_cum *=2
            elif x == 'A3':
                layers += [nn.AvgPool2d(kernel_size=3, stride=2, padding=1)]
                scale_cum *=2
            elif x == 'A4':
                layers += [nn.AvgPool2d(kernel_size=4, stride=4)]
                scale_cum *=4
            else:
                x = int(x * feat_mult)
                if first_layer and input_dim > 64:
                    #scale_cum = 
                    dimo_temp = dimo_temp - 4
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=5, stride=1, padding=0, 
                                             num_classes=num_classes, 
                                             dim_out=dimo_temp, 
                                             first_layer=first_layer)]
                    #dimo_temp = dimo_temp - 2
                    #layers += [LocalLossBlockConv(input_ch, x, kernel_size=3, stride=1, padding=0, 
                    #                         num_classes=num_classes, 
                    #                         dim_out=dimo_temp, 
                    #                         first_layer=first_layer)]
                else:
                    dimo_temp = dimo_temp - 4
                    layers += [LocalLossBlockConv(input_ch, x, kernel_size=5, stride=1, padding=0, 
                                             num_classes=num_classes, 
                                             dim_out=dimo_temp, 
                                             first_layer=first_layer)]
                    #dimo_temp = dimo_temp - 2
                    #layers += [LocalLossBlockConv(input_ch, x, kernel_size=3, stride=1, padding=0, 
                    #                         num_classes=num_classes, 
                    #                         dim_out=dimo_temp, 
                    #                         first_layer=first_layer)]
                input_ch = x
                first_layer = False
        
        return nn.Sequential(*layers), dimo_temp


    
def count_parameters(model):
    ''' Count number of parameters in model influenced by global loss. '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_one_hot(y, n_dims=None):
    ''' Take integer tensor y with n dims and convert it to 1-hot representation with n+1 dims. '''
    y_tensor = y.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot

#编写好阈值训练其余原版SNN，发现bug很可能是这个位置的dim,size问题
def similarity_matrix(x):
    ''' Calculate adjusted cosine similarity matrix of size x.size(0) x x.size(0). '''
    if x.dim() == 4:
        if not args.no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0),-1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc**2, dim=1))).unsqueeze(1)
    R = xn.matmul(xn.transpose(1,0)).clamp(-1,1)
    return R

checkpoint = None
if not args.resume == '':
    if os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume)
        args.model = checkpoint['args'].model
        args_backup = args
        args = checkpoint['args']
        args.optim = args_backup.optim
        args.momentum = args_backup.momentum
        args.weight_decay = args_backup.weight_decay
        args.dropout = args_backup.dropout
        args.no_batch_norm = args_backup.no_batch_norm
        args.cutout = args_backup.cutout
        args.length = args_backup.length
        print('=> loaded checkpoint "{}" (epoch {})'.format(args.resume, checkpoint['epoch']))
    else:
        print('Checkpoint not found: {}'.format(args.resume))
        
if args.model == 'mlp':
    model = Net(args.num_layers, args.num_hidden, input_dim, input_ch, num_classes)
elif args.model.startswith('vgg'):
    model = VGGn(args.model, input_dim, input_ch, num_classes, args.feat_mult)
elif args.model.startswith('snn'):
    model = SNNn(args.model, input_dim, input_ch, num_classes, args.feat_mult)
elif args.model == 'resnet18':
    model = ResNet(BasicBlock, [2,2,2,2], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'resnet34':
    model = ResNet(BasicBlock, [3,4,6,3], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'resnet50':
    model = ResNet(Bottleneck, [3,4,6,3], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'resnet101':
    model = ResNet(Bottleneck, [3,4,23,3], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'resnet152':
    model = ResNet(Bottleneck, [3,8,36,3], num_classes, input_ch, args.feat_mult, input_dim)
elif args.model == 'wresnet10-8':
    model = Wide_ResNet(10, 8, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet10-8a':
    model = Wide_ResNet(10, 8, args.dropout, num_classes, input_ch, input_dim, True)
elif args.model == 'wresnet16-4':
    model = Wide_ResNet(16, 4, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet16-4a':
    model = Wide_ResNet(16, 4, args.dropout, num_classes, input_ch, input_dim, True)
elif args.model == 'wresnet16-8':
    model = Wide_ResNet(16, 8, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet16-8a':
    model = Wide_ResNet(16, 8, args.dropout, num_classes, input_ch, input_dim, True)
elif args.model == 'wresnet28-10':
    model = Wide_ResNet(28, 10, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet28-10a':
    model = Wide_ResNet(28, 10, args.dropout, num_classes, input_ch, input_dim, True)
elif args.model == 'wresnet40-10':
    model = Wide_ResNet(40, 10, args.dropout, num_classes, input_ch, input_dim)
elif args.model == 'wresnet40-10a':
    model = Wide_ResNet(40, 10, args.dropout, num_classes, input_ch, input_dim, True)
else:
    print('No valid model defined')

# Check if to load model
if checkpoint is not None:
    model.load_state_dict(checkpoint['state_dict'])
    args = args_backup
    
if args.cuda:
    model.cuda()

if args.progress_bar:
    from tqdm import tqdm
    
if args.optim == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
elif args.optim == 'adam' or args.optim == 'amsgrad':
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.optim == 'amsgrad')
else:
    print('Unknown optimizer')

model.set_learning_rate(args.lr)
print(model)
print('Model {} has {} parameters influenced by global loss'.format(args.model, count_parameters(model)))

def train(epoch, lr):
    ''' Train model on train set'''
    model.train()
    correct = 0
    loss_total_local = 0
    loss_total_global = 0
    
    # Add progress bar
    if args.progress_bar:
        pbar = tqdm(total=len(train_loader))
        
    # Clear layerwise statistics
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv) or isinstance(m, LocalLossBlockLinearNosnn):
                m.clear_stats()
                
    # Loop train set
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
  
        # Clear accumulated gradient
        optimizer.zero_grad()
        model.optim_zero_grad()
                    
        output, loss = model(data, target, target_onehot)
        loss_total_local += loss * data.size(0)
        loss = F.cross_entropy(output, target)
        if args.loss_sup == 'predsim' and not args.backprop:
            loss *= (1 - args.beta) 
        loss_total_global += loss.item() * data.size(0)
             
        # Backward pass and optimizer step
        # For local loss functions, this will only affect output layer
        loss.backward()
        optimizer.step()
        
        # If special option for no detaching is set, update weights also in hidden layers
        if args.no_detach:
            model.optim_step()
        
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()
        
        # Update progress bar
        if args.progress_bar:
            pbar.set_postfix(loss=loss.item(), refresh=False)
            pbar.update()
            
    if args.progress_bar:
        pbar.close()
        
    # Format and print debug string
    loss_average_local = loss_total_local / len(train_loader.dataset)
    loss_average_global = loss_total_global / len(train_loader.dataset)
    error_percent = 100 - 100.0 * float(correct) / len(train_loader.dataset)
    string_print = 'Train epoch={}, lr={:.2e}, loss_local={:.4f}, loss_global={:.4f}, error={:.3f}%, mem={:.0f}MiB, max_mem={:.0f}MiB\n'.format(
        epoch,
        lr, 
        loss_average_local,
        loss_average_global,
        error_percent,
        torch.cuda.memory_allocated()/1e6,
        torch.cuda.max_memory_allocated()/1e6)
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv) or isinstance(m, LocalLossBlockLinearNosnn):
                string_print += m.print_stats() 
    print(string_print)
    
    return loss_average_local+loss_average_global, error_percent, string_print
                   
def test(epoch):
    ''' Run model on test set '''
    model.eval()
    test_loss = 0
    correct = 0
    
    # Clear layerwise statistics
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv) or isinstance(m, LocalLossBlockLinearNosnn):
                m.clear_stats()
    
    # Loop test set
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        target_ = target
        target_onehot = to_one_hot(target, num_classes)
        if args.cuda:
            target_onehot = target_onehot.cuda()
        
        with torch.no_grad():
            output, _ = model(data, target, target_onehot)
            test_loss += F.cross_entropy(output, target).item() * data.size(0)
        pred = output.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target_).cpu().sum()

    # Format and print debug string
    loss_average = test_loss / len(test_loader.dataset)
    if args.loss_sup == 'predsim' and not args.backprop:
        loss_average *= (1 - args.beta)
    error_percent = 100 - 100.0 * float(correct) / len(test_loader.dataset)
    string_print = 'Test loss_global={:.4f}, error={:.3f}%\n'.format(loss_average, error_percent)
    if not args.no_print_stats:
        for m in model.modules():
            if isinstance(m, LocalLossBlockLinear) or isinstance(m, LocalLossBlockConv) or isinstance(m, LocalLossBlockLinearNosnn):
                string_print += m.print_stats()                
    print(string_print)
    
    return loss_average, error_percent, string_print

''' The main training and testing loop '''
start_epoch = 1 if checkpoint is None else 1 + checkpoint['epoch']
test_minerror = 100
for epoch in range(start_epoch, args.epochs + 1):
    # Decide learning rate
    lr = args.lr * args.lr_decay_fact ** bisect_right(args.lr_decay_milestones, (epoch-1))
    save_state_dict = False
    for ms in args.lr_decay_milestones:
        if (epoch-1) == ms:
            print('Decaying learning rate to {}'.format(lr))
            decay = True
        elif epoch == ms:
            save_state_dict = True

    # Set learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    model.set_learning_rate(lr)
    
    # Check if to remove NClassRandomSampler from train_loader
    if args.classes_per_batch_until_epoch > 0 and epoch > args.classes_per_batch_until_epoch and isinstance(train_loader.sampler, NClassRandomSampler):
        print('Remove NClassRandomSampler from train_loader')
        train_loader = torch.utils.data.DataLoader(dataset_train, sampler = None, batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # Train and test    
    train_loss,train_error,train_print = train(epoch, lr)
    test_loss,test_error,test_print = test(epoch)
    if test_error < test_minerror:
        test_minerror = test_error
    print('Test min error={:.3f}%\n'.format(test_minerror))

    # Check if to save checkpoint
    if args.save_dir is not '':
        # Resolve log folder and checkpoint file name
        filename = 'chkp_ep{}_lr{:.2e}_trainloss{:.2f}_testloss{:.2f}_trainerr{:.2f}_testerr{:.2f}.tar'.format(
                epoch, lr, train_loss, test_loss, train_error, test_error)
        dirname = os.path.join(args.save_dir, args.dataset)
        dirname = os.path.join(dirname, '{}_mult{:.1f}'.format(args.model, args.feat_mult))
        dirname = os.path.join(dirname, '{}_{}x{}_{}_{}_dimdec{}_alpha{}_beta{}_bs{}_cpb{}_drop{}{}_bn{}_{}_wd{}_bp{}_detach{}_lr{:.2e}'.format(
                args.nonlin, args.num_layers, args.num_hidden, args.loss_sup + '-bio' if args.bio else args.loss_sup, args.loss_unsup, args.dim_in_decoder, args.alpha, 
                args.beta, args.batch_size, args.classes_per_batch, args.dropout, '_cutout{}x{}'.format(args.n_holes, args.length) if args.cutout else '', 
                int(not args.no_batch_norm), args.optim, args.weight_decay, int(args.backprop), int(not args.no_detach), args.lr))
        
        # Create log directory
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        elif epoch==1 and os.path.exists(dirname):
            # Delete old files
            for f in os.listdir(dirname):
                os.remove(os.path.join(dirname, f))
        
        # Add log entry to log file
        with open(os.path.join(dirname, 'log.txt'), 'a') as f:
            if epoch == 1:
                f.write('{}\n\n'.format(args))
                f.write('{}\n\n'.format(model))
                f.write('{}\n\n'.format(optimizer))
                f.write('Model {} has {} parameters influenced by global loss\n\n'.format(args.model, count_parameters(model)))
            f.write(train_print)
            f.write(test_print)
            f.write('\n')
            f.close()
        
        # Save checkpoint for every epoch
        torch.save({
            'epoch': epoch,
            'args': args,
            'state_dict': model.state_dict() if (save_state_dict or epoch==args.epochs) else None,
            'train_loss': train_error,
            'train_error': train_error,
            'test_loss': test_loss,
            'test_error': test_error,
        }, os.path.join(dirname, filename))  
    
        # Save checkpoint for last epoch with state_dict (for resuming)
        torch.save({
            'epoch': epoch,
            'args': args,
            'state_dict': model.state_dict(),
            'train_loss': train_error,
            'train_error': train_error,
            'test_loss': test_loss,
            'test_error': test_error,
        }, os.path.join(dirname, 'chkp_last_epoch.tar')) 
   