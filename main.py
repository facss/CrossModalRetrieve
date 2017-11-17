#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torchvision 
import torch.utils.data 
import torch.optim as optim
import trainer 
from torchvision import dataset ,models,transforms
from torch.utils.data import dataset,DataLoader

from model import imvstxt
from dataloader import Data
# created python file
from args import get_parser

#============================================
myparser=get_parser()
opts=myparser.parse_args()
#===========================================

def main():
    #######################################1.data loader###########################################
    train_transforms=transforms.Compose([
        transforms.Scale(256),# rescale the image keeping the original aspect ratio
        transforms.CenterCrop(256), # we get only the center of that rescaled
        transforms.RandomCrop(224), # random crop within the center crop 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    val_transforms=transforms.Compose([
                transforms.Scale(256), # rescale the image keeping the original aspect ratio
                transforms.CenterCrop(224), # we get only the center of that rescaled
                transforms.ToTensor(),
                
    ])
    test_transforms=transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    traindata=Data(opts.train_list_str,opts.test_list_str,opts.val_list_str,
                    train_transforms,None,None,opts.data_path,opts.num_classes)
    valdata=Data(opts.train_list_str,opts.test_list_str,opts.val_list_str,
                    None,val_transforms,None,opts.data_path,opts.num_classes)
    testdata=Data(opts.train_list_str,opts.val_list_str,opts.test_list_str,
                    None,None,test_transforms,opts.data_path,opts.num_classes)
    train_loader=Dataloader(traindata,data_path=opts.data_path,sem_reg=opts.semantic_reg,partition='train',
                        batch_size=opts.batch_size,shuffle=True,num_workers=opts.workers,pin_memory=True)
    print('Training loader prepared')

    val_loader=Dataloader(valdata,data_path=opts.data_path,sem_reg=opts.semantic_reg,partition='val',
                        batch_size=opts.batch_size,shuffle=False,num_workers=opts.workers,pin_memory=True)
    print('Validation loader prepared')

    #testloader=Dataloader(testdata,testloader,)

    ##########################################2.model#################################################
    model=imvstxt()
    model.visionMLP=torch.nn.DataParaller(model.visionMLP,devce_ids=[0,1])
    if opts.cuda:
        model.cuda()

    ########################################3.train && optimer###########################################

    #define loss function (criterion) and optimzer
    #cosine similarity between embeddings ->input1 ,input2,target
    if opts.cuda:
        cosine_crit=nn.CosineEmbeddingLoss(0.1).cuda()
    else:
        cosine_crit=nn.CosineEmbeddingLoss(0.1)

    if opts.semantic_reg:
        weights_class=torch.Tensor(opts.numClasses).fill_(1)
        weights_class[0]=0 # the background class is set to 0, i.e. ignore
        # CrossEntropyLoss combines LogSoftMax and NLLLoss in one single class
        class_crit=nn.CosineEmbeddingLoss(weigth=weights_class).cuda()
        # we will use two different criterion
        criterion=[cosine_crit,class_crit]
    else:
        criterion=cosine_crit

    ##creating different parameter groups
    vision_params=list(map(id,model.visionMLP.parameters()))
    base_params=filter(lambda p:id(p) not in vision_params,model.parameters())

    optim=optim.Adam([
        {'params':base_params},
        {'params':model.visionMLP.parameters(),'lr':opts.lr*opts.freeVision }
    ],lr=opts.lr*opts.freeText)

    #if checkpoint exsit
    if opts.resume:
        if os.path.isfile(opts.resume):
            print("=> loading checkpoint '{}'".format(opts.resume))
            checkpoint=torch.load(opts.resume)
            opts.start_epoch=checkpoint['epoch']
            best_val=checkpoint['best_val']
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(opts.resume,checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(opts.resume))
            best_val=float('inf')
    else:
        best_val=float('inf')

    #model
    trainer=trainer.Trainer(
    cuda=opts.cuda,model=model,optimizer=optim,criterion= criterion,train_loader=train_loader,val_loader= val_loader,max_iter=opts.max_iter
    )
    try:
        trainer.train(best_val)
    except:
        raise


if __name__=="__main__":
    main()