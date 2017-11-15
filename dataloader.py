#! /usr/bin/env python
#-*- coding:UTF-8 -*-
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os

class Data(Dataset):
    #Need a text,which is constructed as "image_filename  text_filename  sound_filename  class"
    def __init__(self,root_path,train_list_str=None,test_list_str=None,val_list_str=None,
                  num_classes,train_transform=None,test_transform=None,*args):
        super(Data,self).__init__()
        self.root_path=root_path#root path of dataset
        self.train_list_str=train_list_str#information list of images,texts,sounds and classes
        self.test_list_str=test_list_str
        self.val_list_str=val_list_str
        self.num_classes=num_classes#class number
        self.train_transform=train_transform

    def getlist(self,list_str):
        #return a  data list
        self.datalist=list()        
        listpathname=os.path.join(self.root_path,list_str)#concatenate the full pathname of info list 
        with open(listpathname,'r') as f:
            for line in f:#every line have four info:image name,text name,sound name,class
                img,txt,sound,classes=line.split()
                img=self.root_path+'images/'+img#return the full path of the image
                txt=self.root_path+'texts/'+txt#return the full path of the text
                sound=self.root_path+'sounds/'+sound#return the full path of the sound
                self.datalist.append((img,txt,sound,classes))#append every line

        return self.datalist
    
    def openmedia(self,filename,train_transform,test_transform,val_transform):
        ###########open image and convert to grayscale
        img=Imge.open(filename[0]).convert("L")
        if train_transform is not None:
            img=self.train_transform(img)
        elif test_transform is not None:
            img=self.test_transform(img)
        elif val_transform is not None:
            img=self.val_transform(img)

        ########open text
        txttmp=''
        with open(filename[1],'r') as f:
            for line in f:
                txttmp=txttmp+' '+line
        txt=txttmp

        #########open sound
        f=wav.open(filename[2],'rb')
        params=f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]#number of channels,(byte)，rate，Sampling points
        strData=f.readframes(nframes)#read the sound
        waveData=np.fromstring(strData,dtype=np.int16)#change the string to int
        sound = waveData*1.0/(max(abs(waveData)))#wave Amplitude normalized

        #############classes
        class_=filename[3]

        return (img,txt,sound,class_)
            
    def __getitem__(self,item):
        #return tuple (image,text,sound)
        if self.train_list_str is not None:
            traininfolist=getlist(self.train_list_str)#get the train file lists
            filename= traininfolist[item]
            tuplelist=openmedia(filename,self.train_transform,None,None)

        elif self.val_list_str is not None:
            valinfolist=getlist(self.val_list_str)#get the validate file lists
            filename=traininfolist[item]
            tuplelist=openmedia(filename,None,None,self.val_transform)
        
        elif self.test_list_str is not None:
            testinfolist=getlist(self.test_list_str)#get the test file lists
            filename=testinfolist[item]
            tuplelist=openmedia(filename,None,self.test_transform,None)
        
        return tuplelist

    def __len__(self):
        return len(self.datalist)