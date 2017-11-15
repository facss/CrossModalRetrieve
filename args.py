#! /usr/bin/env python
#-*- coding:UTF-8 -*-

#args parser
import argparse

def get_parser():
    parse=argparse.ArgumentParser(description="Pytorch Cross Modal Retrieve. ")
    
    #general
    parse.add_argument("--seed",type=int,default=1234)
    parse.add_argument("--cuda",type=bool,default=True)
    
    #data
    parse.add_argument("--img_path",type=str,default="/media/Dataset/CrossModal/Images")##############################
    parse.add_argument("--text_path",type=str,default="/media/Dataset/CrossModal/texts")
    parse.add_argument("--workers",type=int,default=8)

    #model
    parse.add_argument("--batch_size",type=int,default=64)
    parse.add_argument("--snapshots",type=str,default='/snapshots')

    #im2text model
    parse.add_argument("--numClass",type=int,default=13)


    #training
    parse.add_argument("--lr",type=float,default=0.0001)
    parse.add_argument("--momentum",type=float,default=0.9)
    parse.add_argument("--epochs",type=int,default=200)
    parse.add_argument("--resume",type=str,default='')

    #test
    parse.add_argument("--path_results",type =str,default='image')

    #dataset
    parse.add_argument("--maxlen",type=int,default=20)

    return parse