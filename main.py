'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ckpt',default=0, type=int, help='which ckpt to load (default: 0)')
parser.add_argument('--depth',default=0, type=int, help='which network to train (default: 0)')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/shared/hj14/cifar10-dataset', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/shared/hj14/cifar10-dataset', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
if args.depth == 20:
    net = ResNet20()
elif args.depth == 32:
    net = ResNet32()
elif args.depth == 44:
    net = ResNet44()
elif args.depth == 56:
    net = ResNet56()
else:
    print ("wrong depth")
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint {0}-{1}'.format(args.depth, args.ckpt))
    assert os.path.isdir('/shared/hj14/cifar10-dataset/ckpt-resnet{0}'.format(args.depth)), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('/shared/hj14/cifar10-dataset/ckpt-resnet{0}/ckpt-{1}.pth'.format(args.depth, args.ckpt))

    # print(checkpoint)
    # print(checkpoint['net'])
    # for key, v in enumerate(checkpoint['net']):
    #     print (key, v)
    # checkpoint = torch.load('/shared/hj14/cifar10-dataset/ckpt/ckpt.pth')
    
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch < 100:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    elif epoch < 150:
        optimizer = optim.SGD(net.parameters(), lr=args.lr/10, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr/100, momentum=0.9, weight_decay=1e-4)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            # if (batch_idx == 50): #for furthre sparsity testing
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        best_acc = acc

    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('/shared/hj14/cifar10-dataset/ckpt-resnet{0}'.format(args.depth)):
        os.mkdir('/shared/hj14/cifar10-dataset/ckpt-resnet{0}'.format(args.depth))    
    torch.save(state, '/shared/hj14/cifar10-dataset/ckpt-resnet{0}/ckpt-{1}.pth'.format(args.depth,epoch))
        
for epoch in range(start_epoch, min(200,start_epoch+200)):
    train(epoch)
    test(epoch)

# for epoch in range(start_epoch, min(200,start_epoch+1)):
#     test(epoch)