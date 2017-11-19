#! /usr/bin/env python
#-*- coding:UTF-8 -*-
from __future__ import print_function
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import re

class Data(Dataset):
    #Need a text,which is constructed as "image_filename  text_filename  sound_filename  class"
    def __init__(self,train_list_str,test_list_str,val_list_str,train_transform,val_transform,test_transform,root_path,num_classes):
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
                #img,txt,sound,classes=line.split()#three modal
                txt,img,classes=line.split() #two modal
                img=os.path.join(self.root_path,'images',img)+'.jpg'#return the full path of the image             
                txt=os.path.join(self.root_path,'texts',txt)+'.xml'#return the full path of the text
                #sound=self.root_path+'sounds/'+sound+'.wav'#return the full path of the sound
                #self.datalist.append((img,txt,sound,classes))#append every line
                self.datalist.append((img,txt,classes))#append every line

        return self.datalist

    def parserXML(self,XMLname):    
        #only for wikipedia dataset
        if not os.path.exists(XMLname):
	        print ('{} not exists'.format(XMLname))
	        sys.exit(0)
        text=open(XMLname).read()
        beginloc=text.find('<text>')
        endloc=text.find('</text>')
        text_loc=text[beginloc+6:endloc]
        #tree = ET.parse(XMLname) 
        #text= tree.find('text').text
        
        return text_loc
    
    def openmedia(self,filename,train_transform,val_transform,test_transform):
        ###########open image and convert to grayscale
        img=Image.open(filename[0]).convert("L")
        if train_transform is not None:
            img=self.train_transform(img)
        elif test_transform is not None:
            img=self.test_transform(img)
        elif val_transform is not None:
            img=self.val_transform(img)

        ########open text
        txttmp=''
        txt=self.parserXML(filename[1])

        #########open sound ###############
        '''f=wav.open(filename[2],'rb')
        params=f.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]#number of channels,(byte)，rate，Sampling points
        strData=f.readframes(nframes)#read the sound
        waveData=np.fromstring(strData,dtype=np.int16)#change the string to int
        sound = waveData*1.0/(max(abs(waveData)))#wave Amplitude normalized'''

        #############classes
        class_=filename[2] #two modal
        #class_=filename[3] #three modal

        return (img,txt,class_)#two modal
        #return (img,txt,sound,class_)#three modal
            
    def __getitem__(self,item):
        #return tuple (image,text,sound)
        if self.train_list_str is not None:
            traininfolist=self.getlist(self.train_list_str)#get the train file lists
            filename= traininfolist[item]
            tuplelist=self.openmedia(filename,self.train_transform,None,None)

        elif self.val_list_str is not None:
            valinfolist=self.getlist(self.val_list_str)#get the validate file lists
            filename=traininfolist[item]
            tuplelist=self.openmedia(filename,None,self.val_transform,None)
        
        elif self.test_list_str is not None:
            testinfolist=self.getlist(self.test_list_str)#get the test file lists
            filename=testinfolist[item]
            tuplelist=self.openmedia(filename,None,None,self.test_transform)
        
        return tuplelist

    def __len__(self):
        return len(self.datalist)