#! /usr/bin/env python
#-*- coding:UTF-8 -*-

#args parser
import argparse

def get_parser():
    parse=argparse.ArgumentParser(description="Pytorch Cross Modal Retrieve. ")
    
    #general
    parse.add_argument("--seed",type=int,default=1234)
    parse.add_argument("--cuda",type=bool,default=True,help='whether the GPU is available')
    
    #data
    parse.add_argument("--data_path",type=str,default='/media/Dataset/project',help='root path of data')
    parse.add_argument("--img_path",type=str,default="/media/Dataset/project/images")##############################
    parse.add_argument("--text_path",type=str,default="/media/Dataset/project/texts")
    parse.add_argument("--sound_path",type=str,default="/media/Dataset/project/sounds")
    parse.add_argument("--train_list_str",type=str,default='/media/Dataset/project/trainlist.txt',help='full path to trainlist.txt')
    parse.add_argument("--test_list_str",type=str,default='/media/Dataset/project/testlist.txt',help='full path to testlist.txt')
    parse.add_argument("--val_list_str",type=str,default='/media/Dataset/project/vallist.txt',help='full path to vallist.txt')
    parse.add_argument("--workers",type=int,default=8)

    #model
    parse.add_argument("--batch_size",type=int,default=64)
    parse.add_argument("--snapshots",type=str,default='/snapshots')

    #imvstxt model
    parse.add_argument("--num_classes",type=int,default=13)
    parse.add_argument("--embDim",type=int,default=1024)
    parse.add_argument('--crnnDim', default=1024, type=int)
    parse.add_argument('--srnnDim', default=300, type=int)
    parse.add_argument('--imfeatDim', default=2048, type=int)
    parse.add_argument('--contextDim', default=1024, type=int)
    parse.add_argument('--segmentW2VDim', default=300, type=int)
    parse.add_argument('--maxSeqlen', default=20, type=int)
    parse.add_argument('--maxIngrs', default=20, type=int)
    parse.add_argument('--maxImgs', default=5, type=int)
    parse.add_argument('--numClasses', default=1048, type=int)
    parse.add_argument('--preModel', default='resNet50',type=str)
    parse.add_argument('--semantic_reg', default=True,type=bool)


    #training
    parse.add_argument("--lr",type=float,default=0.0001,help='learning rate')
    parse.add_argument("--momentum",type=float,default=0.9,help='SGD momentum')
    parse.add_argument('--freeVision', default=False, type=bool)
    parse.add_argument('--freeText', default=True, type=bool)
    parse.add_argument('--segmentW2V', default='data/vocab.bin',type=str)
    parse.add_argument("--epochs",type=int,default=500,help='number of epochs')
    parse.add_argument("--start_epoch",type=int,default=100,help='the start epoch position of checkpoint')
    parse.add_argument("--max_iter",type=int,default=100000,help='max number of iterations')
    parse.add_argument("--cos_weight",type=float,default=0.98,help='')
    parse.add_argument("--cls_wight",type=float,default=0.01,help='')
    parse.add_argument('--valfreq', default=10,type=int)  
    parse.add_argument('--patience', default=1, type=int)
    parse.add_argument("--resume",type=str,default='')

    #test
    parse.add_argument("--path_results",type =str,default='image')

    #dataset
    parse.add_argument("--maxlen",type=int,default=20)

    return parse
