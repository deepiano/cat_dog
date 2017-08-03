"""
[source] 
imagenet example - https://github.com/pytorch/examples/blob/master/imagenet/main.py
densenet - 

"""
import argparse
import os
import shutil
import time
import pdb

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
#import torchvision.models as models

import cv2
import numpy as np

import my_densenet 
#import my_transform

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

data_root = '/mnt/nfs/bong/MyProject/cat_dog/data'

best_prec1 = 0


def main():

    global args, best_prec1
    args = parser.parse_args()

    arch = 'densenet121'
    print("=> creating model '{}'".format(arch))
#    model = my_resnet.resnet34(pretrained=False, num_classes=2) 
    model = my_densenet.densenet121(pretrained=False, num_classes=2)
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.CrossEntropyLoss().cuda()
#    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), 0.1,
                                momentum=0.9,
                                weight_decay=1e-4)

   # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    
#   Data loading code
    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    testdir = os.path.join(data_root, 'train')

    mean = [106.2072 / 255, 115.9283 / 255, 124.4055 / 255]
    std = [65.5968 / 255, 64.9490 / 255, 66.6102 / 255]
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
                            transforms.Scale(256),
                            transforms.RandomSizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize])

    train_data = datasets.ImageFolder(traindir,\
                                transform=train_transform)

    train_loader = torch.utils.data.DataLoader(\
                    train_data,\
                    batch_size=200,\
                    shuffle=True,\
                    num_workers=8,\
                    pin_memory=True)
    
    val_transform = transforms.Compose([\
                        transforms.Scale(224),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize])

    val_data = datasets.ImageFolder(valdir,\
                            transform=val_transform)
    
    val_loader = torch.utils.data.DataLoader(\
                    val_data,\
                    batch_size=100,\
                    shuffle=False,\
                    num_workers=8,\
                    pin_memory=True)

    test_transform = transforms.Compose([
                            transforms.Scale(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])

    test_data = datasets.ImageFolder(testdir,\
                                transform=test_transform)

    test_loader = torch.utils.data.DataLoader(\
                    test_data,\
                    batch_size=200,\
                    shuffle=False,\
                    num_workers=8,\
                    pin_memory=True)

#    for epoch in range(1):
#        test(train_loader, model, criterion)
#
#    exit()

    # ensemble
    models = []
    models.append(model)
    best_precs = [0,]
    predictions = []

    for i in range(9):

        model_ = my_densenet.densenet121(pretrained=False, num_classes=2)
        model_ = torch.nn.DataParallel(model_).cuda()
        models.append(model_)
        best_precs.append(0)

    for epoch in range(5):

        adjust_learning_rate(optimizer, epoch)

        for i, m in enumerate(models):
   
            filename = './checkpoint/checkpoint' + str(i) + '.pth.tar'
            model_filename = './model_best/model_best' + str(i) + '.pth.tar'
    
            # train for one epoch
            train(train_loader, m, criterion, optimizer, epoch)
    
            # evaluate on validation set
            prec = validate(val_loader, m, criterion)
    
            is_best = prec > best_precs[i]
            best_prec = max(prec1, best_precs[i])
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': m.state_dict(),
                'best_prec1': best_prec,
                'optimizer' : optimizer.state_dict(),
                },\
                is_best,\
                filename=filename,\
                model_filename=model_filename)

#            m.eval()
#        
#            for i, (input, target) in enumerate(val_loader):
#        
#                target = target.cuda(async=True)
#                input_var = torch.autograd.Variable(input, volatile=True)
#                target_var = torch.autograd.Variable(target, volatile=True)
#        
#                # compute output
#                output = m(input_var)
#        
#                _, pred = torch.max(output.data, 1)
#                
#
#                # measure accuracy and record loss
#                prec1 = accuracy(output.data, target)
#                losses.update(loss.data[0], input.size(0))
#                top1.update(prec1, input.size(0))
        

#    for i, (input, target) in enumerate(train_loader):
     
#        print ('input : ',  i, input)
#        print ('target : ',  i, target)
#        input_ary = input.numpy()
#        input_sqz = np.squeeze(input_ary).T
#        input_shape = input_sqz.shape
#        print('shape : ', input_shape)  
#        input_rot = np.rot90(input_sqz, 3) 
#        print('shape : ', input_rot.shape)  
#        cv2.imshow("cat", input_rot)
#        cv2.waitKey()
#        cv2.destroyAllWindows()
        

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter() 
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print_freq = 90 
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    epoch, i, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))
        
        



def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
#    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):

#        input_ary = input.numpy()
#        input_sqz = np.squeeze(input_ary).T
#        input_shape = input_sqz.shape
#        print(input_shape) 
#        exit() 

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)


        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
#        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
#        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print_freq = 40 
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f} '
          .format(top1=top1))

    return top1.avg


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # hard example mining
#        _, pred = torch.max(output.data, 1)
#        equal_mask = pred.eq(target).cpu().numpy()
#        false_index = np.where(equal_mask == False)[0]
        
#        for i in false_index:
#            print('index : ', i)
#            print('output: ', output[i])
#            print('target: ', target[i])
#            print('image path: ', test_loader.dataset.imgs[i])
#            pdb.set_trace()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print_freq = 25 
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   i, len(test_loader), batch_time=batch_time, loss=losses,
                   top1=top1))

    print(' * Prec@1 {top1.avg:.3f} '
          .format(top1=top1))

    return top1.avg


def save_checkpoint(state, is_best,\
                    filename='checkpoint.pth.tar',\
                    model_filename='model_best.pth.tar'):

    torch.save(state, filename)
    if is_best:
        #shutil.copyfile(filename, 'model_best.pth.tar')
        shutil.copyfile(filename, model_filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#    lr = 0.1 * (0.1 ** (epoch // 30))   
    lr = 0.1 * (0.1 ** (epoch // 30))   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target):
    """Computes the accuracy of output"""

    batch_size = target.size(0)

    _, pred = torch.max(output, 1)
    correct = pred.eq(target).sum()
    res = correct * 100 / float(batch_size)

    return res


if __name__=='__main__':

    main()









