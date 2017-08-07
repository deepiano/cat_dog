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
import csv

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np

import my_densenet
import my_resnet

data_root = './data'

class MyImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        return super(MyImageFolder, self).__getitem__(index), self.imgs[index]

def main():

    models = []
    for i in range(9):
        if i < 3:
            model = my_densenet.densenet121(pretrained=False, num_classes=2)
        elif i == 3 or i == 4:
            model = my_densenet.densenet_mid(pretrained=False, num_classes=2)
        elif i == 5 or i == 6:
            model = my_densenet.densenet_small(pretrained=False, num_classes=2)
        else:
            model = my_resnet.resnet34(pretrained=False, num_classes=2)

        model = torch.nn.DataParallel(model).cuda()
        model_path = './model_best/model_best' + str(i) + '.pth.tar'
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])

        models.append(model)

    cudnn.benchmark = True
    
#   Data loading code
    testdir = os.path.join(data_root, 'test')

    mean = [106.2072 / 255, 115.9283 / 255, 124.4055 / 255]
    std = [65.5968 / 255, 64.9490 / 255, 66.6102 / 255]
    normalize = transforms.Normalize(mean=mean, std=std)

    test_transform = transforms.Compose([
                            transforms.Scale(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize])

    test_data = MyImageFolder(testdir,\
                                transform=test_transform)

    test_loader = torch.utils.data.DataLoader(\
                    test_data,\
                    batch_size=200,\
                    shuffle=False,\
                    num_workers=8,\
                    pin_memory=True)


def test(test_loader, model, criterion):

    prec = AverageMeter()

    cvsfile = open('./result/result.csv', 'w')
    fieldnames = ['filename', 'lable']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()


    for i, data in enumerate(test_loader):

        (input, target), (filepath, index) = data

        batch_size = input.size(0)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        vote_sum = np.zeros([batch_size, 2])

        for model_index, model in enumerate(models):

            model.eval()

            # compute output
            output = model(input_var)
            _, pred = torch.max(output.data, 1)
            pred = pred.cpu().numpy()

            for img_index in range(batch_size):
                vote_sum[img_index][pred[img_index]] += 1

        for img_index in range(batch_size):
            label = -1
            if vote_sum[img_index][0] < vote_sum[img_index][1]:
                label = 1
            else:
                label = 0
            filename = os.path.basename(filepath)
            writer.writerow({'filename':filename, 'label':label}) 




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


def accuracy(output, target):
    """Computes the accuracy of output"""

    batch_size = target.size(0)

    _, pred = torch.max(output, 1)
    correct = pred.eq(target).sum()
    res = correct * 100 / float(batch_size)

    return res


if __name__=='__main__':

    main()









