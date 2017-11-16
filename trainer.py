#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
import time
import numpy as np
import torch.autograd.Variable as Variable

from model import imvstxt
from args import get_parser

#=====================================================
parse=get_parser()
opts=parse.parse_args()
#=====================================================
def adjust_learning_rate(optimizer,epoch,opts):
    """Switching between modalities"""
    # parameters corresponding to the rest of the network
     optimizer.param_groups[0]['lr'] = opts.lr * opts.freeRecipe
    # parameters corresponding to visionMLP 
    optimizer.param_groups[1]['lr'] = opts.lr * opts.freeVision 

    print 'Initial base params lr: %f' % optimizer.param_groups[0]['lr']
    print 'Initial vision lr: %f' % optimizer.param_groups[1]['lr']

    # after first modality change we set patience to 3
    opts.patience = 3

def accuracy(output,target,topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk=max(topk)
    batch_size=target.size(0)

    _,pred=output.topk(maxk,1,True,True)
    pred=pred.t()
    correct=pred.eq(target.view(1,-1).expand_as(pred))

    res=[]
    for k in topk:
        correct_k=correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res

def save_checkpoint(state,is_best,filename='checkpoint.path.tar'):
    torch.save(state,filename)
    if is_best:
        shutil.copy(filename,'model_best.pth.tar')

class AverageMeter(object):
    """Compute and store the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg=self.sum/self.count

def rank(opts,im_embeds,rec_embeds,rec_ids):
    random.seed(opts.seed)
    type_embedding = opts.embtype 
    im_vecs = img_embeds 
    instr_vecs = rec_embeds 
    names = rec_ids

    # Sort based on names to always pick same samples for medr
    idxs = np.argsort(names)
    names = names[idxs]

    # Ranker
    N = opts.medr
    idxs = range(N)

    glob_rank = []
    glob_recall = {1:0.0,5:0.0,10:0.0}
    for i in range(10):

        ids = random.sample(xrange(0,len(names)), N)
        im_sub = im_vecs[ids,:]
        instr_sub = instr_vecs[ids,:]
        ids_sub = names[ids]

        # if params.embedding == 'image':
        if type_embedding == 'image':
            sims = np.dot(im_sub,instr_sub.T) # for im2recipe
        else:
            sims = np.dot(instr_sub,im_sub.T) # for recipe2im

        med_rank = []
        recall = {1:0.0,5:0.0,10:0.0}

        for ii in idxs:

            name = ids_sub[ii]
            # get a column of similarities
            sim = sims[ii,:]

            # sort indices in descending order
            sorting = np.argsort(sim)[::-1].tolist()

            # find where the index of the pair sample ended up in the sorting
            pos = sorting.index(ii)

            if (pos+1) == 1:
                recall[1]+=1
            if (pos+1) <=5:
                recall[5]+=1
            if (pos+1)<=10:
                recall[10]+=1

            # store the position
            med_rank.append(pos+1)

        for i in recall.keys():
            recall[i]=recall[i]/N

        med = np.median(med_rank)
        # print "median", med

        for i in recall.keys():
            glob_recall[i]+=recall[i]
        glob_rank.append(med)

    for i in glob_recall.keys():
        glob_recall[i] = glob_recall[i]/10

    return np.average(glob_rank), glob_recall
    
class Trainer(object):
    def __init__(self,cuda,model,optimizer,criterion,train_loader,val_loader,max_iter):
        self.cuda=cuda
        self.model=model
        self.optim=optimizer
        self.train_loader=train_loader
        self.val_loader=val_loader
        self.epochs=0
        self.iteration=0
        self.top1=AverageMeter()
        self.losses=AverageMeter()
        self.criterion=criterion


    def validate(self,val_loader,model,criterion):
        batch_time=AverageMeter()
        cos_loss=AverageMeter()
        if opts.semantic_reg:
            img_losses=AverageMeter()
            rec_losses=AverageMeter()

        #switch to evaluate mode
        model.eval()
        end=time.time()
        for i,(input,target) in enumerate(val_loader):
            input_var=list()
            for j in range(len(input)):
                input_var.append(Variable(input[j],volatile=True).cuda())
            target_var=list()
            for j in range(len(target)-2): #we do not consider the last two object of the list
                target[j]=target[j].cuda(async=True)
                target_var.append(Variable(target[j],volatile=True))

            #compute output
            output=model(input_var[0],input_var[1],input_var[2],input_var[3],input_var[4])

            if i==0:
                data0=output[0].data.cpu().numpy()
                data1=output[1].data.cpu().numpy()
                data2=target[-2]
                data3=target[-1]
            else:
                data0=np.concatenate((data0,output[0].data.cpu().numpy()),axis=0)
                data1=np.concatenate((data1,output[1].data.cpu().numpy()),axis=0)
                data2=np.concatenate((data2,target[-2]),axis=0)
                data3=np.concatenate((data3,target[-1]),axis=0)
        
        medR, recall = rank(opts, data0, data1, data2)
        print('* Val medR {medR:.4f}\t'
          'Recall {recall}'.format(medR=medR, recall=recall))

        return medR 


    def train_epoch(self,train_loader, model, criterion, optimizer, epoch):
        batch_time=AverageMeter()
        data_time=AverageMeter()
        cos_losses=AverageMeter()
        if opts.semantic_reg:
            img_losses=AverageMeter()
            rec_losses=AverageMeter()
        top1=AverageMeter()
        top5=AverageMeter()

        #switch to train mode
        model.train()

        end=time.time()
        for i ,(input,target) in enumerate(train_loader):
            #measure data loading time
            data_time.update(time.time()-end)

            input_val=list()
            for j in range(len(input)):
                input_var.append(torch.autograd.Variable(input[j]).cuda())

            target_var=list()
            for j in range(len(target)):
                target[j]=target[j].cuda(async=True)
                target_var.append(torch.autograd.Variable(target[j]))

            #compute output
            output=model(input_var[0],input_var[1],input_var[2],input_var[3],input_var[4])

            #compute loss
            if opts.semantic_reg:
                cos_loss=criterion[0](output[0],output[1],target_var[0])
                img_loss=criterion[1](output[2],target_var[1])
                rec_loss=criterion[1](output[3],target_var[2])
                #combined loss
                loss=opts.cos_weight*cos_loss+\
                     opts.cls_weight*img_loss+\
                     opts.cls_weigth*rec_loss

                #measure performance and record losses
                cos_losses.update(cos_loss.data[0],input[0].size(0))
                img_losses.update(img_loss.data[0],input[0].size(0))

            else:
                loss=criterion(output[0],output[1],target_var[0])
                #measure performance and record loss
                cos_losses.update(loss.data[0],input[0].size(0))
            #compute gradient and do Adam step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #measure elapsed time
            batch_time.update(time.time()-end)
            end=time.time()
        if opts.semantic_reg:
            print('Epoch: {0}\t'
                  'cos loss {cos_loss.val:.4f} ({cos_loss.avg:.4f})\t'
                  'img Loss {img_loss.val:.4f} ({img_loss.avg:.4f})\t'
                  'rec loss {rec_loss.val:.4f} ({rec_loss.avg:.4f})\t'
                  'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
                   epoch, cos_loss=cos_losses, img_loss=img_losses,
                   rec_loss=rec_losses, visionLR=optimizer.param_groups[1]['lr'],
                   recipeLR=optimizer.param_groups[0]['lr']))
        else:
            print('Epoch: {0}\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'vision ({visionLR}) - recipe ({recipeLR})\t'.format(
                   epoch, loss=cos_losses, visionLR=optimizer.param_groups[1]['lr'],
                   recipeLR=optimizer.param_groups[0]['lr'])) 


    def train(self,best_val):

        for epoch in range(opts.start_epoch,opts.epochs):
            
            #train  for one epoch
            train_epoch(self.train_loader,self.model,self.criterion,self.optim,epoch)

            if (epoch+1) %opts.valfreq==0 and epoch !=0:
                val_loss=validate(self.val_loader,self.model,self.criterion)

                #check patience
                if val_loss>=best_val:
                    valtrack+=1
                else:
                    valtrack=0

                if valtrack>=opts.patience:
                    #we swich modalities
                    opts.freeVision=opts.freeText;opts.freeText=not(opts.freeVision)
                    #change the learning rate accordingly
                    adjust_learning_rate(self.optim,epoch,opts)
                    valtrack=0
                
                #save the best model
                is_best = val_loss < best_val
                best_val = min(val_loss, best_val)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_val': best_val,
                    'optimizer': optimizer.state_dict(),
                    'valtrack': valtrack,
                    'freeVision': opts.freeVision,
                    'curr_val': val_loss,
                }, is_best)

                 print '** Validation: %f (best) - %d (valtrack)' % (best_val, valtrack)
