#! /usr/bin/env python
#-*- coding:UTF-8 -*-


from torch.utils.data import dataset,DataLoader
from dataloader import Data
from torchvision import transforms
# created python file
from args import get_parser

#============================================
myparser=get_parser()
opts=myparser.parse_args()
#===========================================
train_transforms=transforms.Compose([
        transforms.Scale(256),# rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256), # we get only the center of that rescaled
        transforms.RandomCrop(224), # random crop within the center crop 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
traindata=Data(opts.train_list_str,None,None,
                    train_transforms,None,None,opts.data_path,opts.num_classes)

testdata=traindata[1]
print(testdata)
