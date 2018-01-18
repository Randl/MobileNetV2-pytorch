import argparse
import os
import shutil
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.optim as optim
import torch.utils.data
from datetime import datetime
from torch.autograd import Variable
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from model import MobileNet2

parser = argparse.ArgumentParser(description='ShuffleNet training with PyTorch')
parser.add_argument('--dataroot', required=True, help='Path to ImageNet train and val folders, preprocessed '
                                                      'as described in '
                                                      'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='torch.cuda.FloatTensor', help='Type of tensor - e.g torch.cuda.HalfTensor')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.045, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.98, help='LR is multiplied by gamma each epoch.')

parser.add_argument('-e', '--evaluate', type=str, metavar='FILE', help='evaluate model FILE on validation set')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')
# Architecture
parser.add_argument('--groups', type=int, default=8, metavar='g', help='Number of groups in ShuffleNet (default 8).')
parser.add_argument('--scaling', type=float, default=0.25, metavar='s', help='Scaling of ShuffleNet (default x0.25).')

args = parser.parse_args()


def save_checkpoint(state, is_best, filepath='./', filename='checkpoint.pth.tar'):
    save_path = os.path.join(filepath, filename)
    best_path = os.path.join(filepath, 'model_best.pth.tar')
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, best_path)


__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def inception_preproccess(input_size, normalize=__imagenet_stats):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize)
    ])


def scale_crop(input_size, scale_size=None, normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    if scale_size != input_size:
        t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def get_transform(augment=True):
    normalize = __imagenet_stats
    scale_size = 256
    input_size = 224
    if augment:
        return inception_preproccess(input_size=input_size, normalize=normalize)
    else:
        return scale_crop(input_size=input_size, scale_size=scale_size, normalize=normalize)


val_data = datasets.ImageFolder(root=os.path.join(args.dataroot, 'val'), transform=get_transform(False))
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                         pin_memory=True)

train_data = datasets.ImageFolder(root=os.path.join(args.dataroot, 'train'), transform=get_transform())
train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                           num_workers=args.workers, pin_memory=True)


def main():
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.evaluate:
        args.results_dir = '/tmp'
    if args.save is '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if 'cuda' in args.type:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        args.gpus = None

    # running on ImageNet
    model = MobileNet2(scale=args.scaling)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print('number of parameters: {}'.format(num_parameters))
    print(model)

    # define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', torch.nn.CrossEntropyLoss)()
    criterion.type(args.type)
    model.type(args.type)

    optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate, alpha=0.9, momentum=args.momentum,
                                    weight_decay=args.decay)

    best_test = 0
    train_acc = []
    test_acc = []

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch'] - 1
            best_test = checkpoint['best_prec1']
            train_acc = checkpoint['train_acc']
            test_acc = checkpoint['test_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.gpus and len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)

    if args.evaluate:
        test(model, criterion)
        return
    for epoch in trange(args.start_epoch, args.epochs + 1):
        args.learning_rate *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate
        train_accuracy = train(model, epoch, optimizer, criterion)
        test_accuracy = test(model, criterion)

        train_acc.append(train_accuracy)
        test_acc.append(test_accuracy)
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict(), 'train_acc': train_acc, 'test_acc': test_acc},
                        test_accuracy > best_test, filepath=save_path)
        if test_accuracy > best_test:
            best_test = test_accuracy

        tqdm.write('Train: \n[{}]\nVal:\n[{}]\n'.format(', '.join('{:.5}'.format(x) for x in train_acc),
                                                        ', '.join('{:.5}'.format(x) for x in test_acc)))  # Dirty


def train(model, epoch, optimizer, criterion):
    model.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        if args.gpus is not None:
            data, target = data.cuda(async=True), target.cuda(async=True)

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        curr_correct = pred.eq(target.data.view_as(pred)).cpu().sum()
        correct += curr_correct

        if batch_idx % args.log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Top-1 accuracy: {:.2f}%({:.2f})%.'.format(epoch, batch_idx, len(train_loader),
                                                           100. * batch_idx / len(train_loader), loss.data[0],
                                                           100. * curr_correct / args.batch_size,
                                                           100. * correct / (args.batch_size * (batch_idx + 1))))
    return correct / len(train_loader.dataset)


def test(model, criterion):
    model.eval()
    test_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
        if args.gpus is not None:
            data, target = data.cuda(async=True), target.cuda(async=True)
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(val_loader.dataset)

    tqdm.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return correct / len(val_loader.dataset)


if __name__ == '__main__':
    main()
