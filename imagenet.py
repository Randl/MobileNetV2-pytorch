import argparse
import os
import shutil
import random
import sys

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.optim as optim
import torch.utils.data
from datetime import datetime
from torchvision import datasets, transforms
from tqdm import tqdm, trange

from logger import CsvLogger
from model import MobileNet2

parser = argparse.ArgumentParser(description='ShuffleNet training with PyTorch')
parser.add_argument('--dataroot', required=True, help='Path to ImageNet train and val folders, preprocessed '
                                                      'as described in '
                                                      'https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset')
parser.add_argument('--gpus', default=None, help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='Number of data loading workers (default: 4)')
parser.add_argument('--type', default='float32',
                    help='Type of tensor: float32, float16, float64, int8. Default: float32')

# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default=96, type=int, metavar='N', help='mini-batch size (default: 96)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.01, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=4e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')

parser.add_argument('-e', '--evaluate', type=str, metavar='FILE', help='evaluate model FILE on validation set')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='Directory to store results')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, metavar='S', help='random seed (default: 1)')

# Architecture
parser.add_argument('--scaling', type=float, default=1, metavar='s', help='Scaling of MobileNet (default x1).')
parser.add_argument('--input_size', type=float, default=224, metavar='s', help='Input size of MobileNet (default 224).')

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


def get_transform(augment=True, input_size=224):
    normalize = __imagenet_stats
    scale_size = 256
    if augment:
        return inception_preproccess(input_size=input_size, normalize=normalize)
    else:
        return scale_crop(input_size=input_size, scale_size=scale_size, normalize=normalize)


val_data = datasets.ImageFolder(root=os.path.join(args.dataroot, 'val'),
                                transform=get_transform(False, args.input_size))
val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                         pin_memory=True)

train_data = datasets.ImageFolder(root=os.path.join(args.dataroot, 'train'),
                                  transform=get_transform(input_size=args.input_size))
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

    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'
    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    elif args.type == 'int8':
        dtype = torch.int8
    else:
        return  # TODO
    # running on ImageNet
    model = MobileNet2(input_size=args.input_size, scale=args.scaling)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    print('number of parameters: {}'.format(num_parameters))
    print(model)

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    model = torch.nn.DataParallel(model, args.gpus)
    model.to(device=device, dtype=dtype)
    criterion.to(device=device, dtype=dtype)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)

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


    if args.evaluate:
        test_loss, test_accuracy = test(model, criterion, device, dtype)  #TODO
        return

    csv_logger = CsvLogger(filepath=save_path)
    csv_logger.save_params(sys.argv, args)

    for epoch in trange(args.start_epoch, args.epochs + 1):
        if epoch in args.schedule:
            args.learning_rate *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.learning_rate
        train_loss, train_accuracy = train(model, epoch, optimizer, criterion, device, dtype)
        test_loss, test_accuracy = test(model, criterion, device, dtype)
        csv_logger.write({'epoch': epoch + 1, 'val_error1': 1 - test_accuracy, 'val_loss': test_loss,
                          'train_error1': 1 - train_accuracy, 'train_loss': train_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_test,
                         'optimizer': optimizer.state_dict(), 'train_acc': train_acc, 'test_acc': test_acc},
                        test_accuracy > best_test, filepath=save_path)
        csv_logger.plot_progress_err(claimed_acc=0.718)  #TODO
        if test_accuracy > best_test:
            best_test = test_accuracy


def train(model, epoch, optimizer, criterion, device, dtype):
    model.train()

    correct = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        curr_correct = pred.eq(target.data.view_as(pred)).cpu().sum().item()
        correct += curr_correct

        if batch_idx % args.log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Top-1 accuracy: {:.2f}%({:.2f})%.'.format(epoch, batch_idx, len(train_loader),
                                                           100. * batch_idx / len(train_loader), loss.item(),
                                                           100. * curr_correct / args.batch_size,
                                                           100. * correct / (args.batch_size * (batch_idx + 1))))
    return  loss.item(), correct / len(train_loader.dataset)


def test(model, criterion, device, dtype):
    model.eval()
    test_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(tqdm(val_loader)):
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        with torch.no_grad():
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss /= len(val_loader)

    tqdm.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    return test_loss, correct / len(val_loader.dataset)


if __name__ == '__main__':
    main()
