'''Train CIFAR10 with PyTorch.'''
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from mup.coord_check import get_coord_data, plot_coord_data
from mup import MuAdam, MuSGD, get_shapes, make_base_shapes, set_base_shapes

import resnet


def coord_check(mup, lr, optimizer, nsteps, arch, base_shapes, nseeds, device='cuda', plotdir='', legend=False):

    optimizer = optimizer.replace('mu', '')

    def gen(w, standparam=False):
        def f():
            model = getattr(resnet, arch)(wm=w).to(device)
            if standparam:
                set_base_shapes(model, None)
            else:
                set_base_shapes(model, base_shapes)
            return model
        return f

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='../dataset', train=True, download=True, transform=transform_train)
    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=1, shuffle=False)

    widths = 2**np.arange(-2., 2)
    models = {w: gen(w, standparam=not mup) for w in widths}
    df = get_coord_data(models, dataloader, mup=mup, lr=lr, optimizer=optimizer, nseeds=nseeds, nsteps=nsteps)

    prm = 'μP' if mup else 'SP'
    plot_coord_data(df, legend=legend,
        save_to=os.path.join(plotdir, f'{prm.lower()}_{arch}_{optimizer}_coord.png'),
        suptitle=f'{prm} {arch} {optimizer} lr={lr} nseeds={nseeds}',
        face_color='xkcd:light grey' if not mup else None)


# Training
def train(epoch, net):
    from utils import progress_bar
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
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


def test(epoch, net):
    from utils import progress_bar
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
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
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc



if __name__ == '__main__':
    base_shapes = get_shapes(net)
    delta_shapes = get_shapes(getattr(resnet, args.arch)(wm=args.width_mult/2))

    net = net.to(device)
    
    set_base_shapes(net, args.load_base_shapes)
    
    criterion = nn.CrossEntropyLoss()
    if args.optimizer == 'musgd':
        optimizer = MuSGD(net.parameters(), lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
    elif args.optimizer == 'muadam':
        optimizer = MuAdam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise ValueError()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)


    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch, net)
        test(epoch, net)
        scheduler.step()