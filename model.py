#! /usr/bin/env python
#-*- coding:UTF-8 -*-


import torch
import torch.nn as nn

# created python file
from args import get_parser
from dataloader import Data
import torhvision.models as models
import torchwordemb

#============================================
myparser=get_parser()
opts=myparser.parse_args()
#===========================================

class TableModule(nn.Module):
    def __init__(self):
        super(TableModule,self).__init__()

    def forward(self,x,dim):
        y=torch.cat(x,dim)
        return y

def norm(input,p=2,dim=1,eps=1e-12):
    return input/input.norm(p,dim,keepdim=True).clamp(min=eps).expand_as(input)

class contextRNN(nn.Module):
    #text which has long context 
    def __init__(self):
        super(contextRNN,self).__init__()
        self.lstm=nn.LSTM(input_size=opts.contextDim,hidden_size=opts.crnnDim,bidirectional=False,batch_first=True)
    
    def forward(self,x,sq_lengths):
        #here we use a previous LSTM to get the representation of each text
        #sort sequence according to the length
        sorted_len,sorted_idx=sq_lengths.sort(0,descending=True)
        index_sorted_idx=sorted_idx\
                  .view(-1,1,1).expand_as(x)
        sorted_inputs=x.gather(0,index_sorted_idx.long())
        #pack sequence
        packed_seq=torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs,sorted_len.spu().data.numpy(),batch_first=True)
        #pass it to the lstm
        out,hidden=self.lstm(packed_seq)

        #unsort the output
        _,original_idx=sorted_idx.sort(0,descending=False)

        unpacked,_=torch.utils.rnn.pad_packed_sequence(out,batch_first=True)
        unsorted_idx=original_idx.view(-1,1,1)/.expand_as(unpacked)
        #we get the last index of each sequence in the batch
        idx =(sq_lengths-1).view(-1,1).expand(unpacked.size(0),unpacked.size(2)).unsequeeze(1)
        #we sort and get the last element of each sequence
        output=unpacked.gather(0,unsorted_idx.long()).gather(1,idx.long())
        output=output.view(output.size(0),output.size(1)*output.size(2))

        return output 

class segmentRNN(nn.Module):
    #text which is segment
    def __init__(self):
        super(segmentRNN,self).__init__()
        self.segnn=nn.LSTM(input_size=opts.segmentW2VDim,hidden_size=opts.srnnDim,bidirectional=True,batch_first=True)
        _,vec=torchwordemb.load_word2vec_bin(opts.segW2V)
        self.embs=nn.Embedding(vec.size(0),opts.segmentW2VDim,padding_idx=0) #not sure about the padding idx
        self.embs.weight.data.copy_(vec)
    
    def forward(self,x,sq_lengths):

        #we get the w2v for each element of the segment sequence
        x=self.embs(x)

        #sort sequence according to the length
        sorted_len,sorted_idx=sq_lengths.sort(0,descending=True)
        index_sorted_idx=sorted_idx\
                 .view(-1,1,1).expand_as(x)
        sorted_inputs=x.gather(0,index_sorted_idx.long())
        #pack sequence
        packed_seq=torch.nn.utils.rnn.pack_padded_sequence(
            sorted_inputs,sorted_len.cpu().data.numpy(),batch_first=True)
        #pass it to the rnn
        out,hidden=self.irnn(packed_seq)

        #unsort the output
        _,original_idx=sorted_idx.sort(0,descending=False)
        
        #LSTM
        #bi-directional
        unsorted_idx=original_idx.view(1,-1,1).expand_as(hidden[0])
        #2 directions x batch_size x num features,we transpose 1st and 2nd dimension
        output=hidden[0].gather(1,unsorted_idx).transpose(0,1).contiguous()
        output=output.view(output.size(0),output.size(1)*output.size(2))

        return output


class imvstxt(nn.Module):
    def __init__(self):
        super(imvstxt,self).__init__()
        if opts.preModel='resNet50':
            resnet=models.resnet50(pretrained=True)
            moduls=list(resnet.children())[:-1]#we dont use the last layer
            self.visionMLP=nn.Sequential(*models)

            self.visual_embedding=nn.Sequential(
                nn.Linear(opts.imfratDim,opts.embDim),
                nn.Tanh(),
            )

            self.txt_embedding=nn.Sequential(
                nn.Linear(opts.srnnDim*2+opts.crnnDim,opts.embDim,opts.embDim),
                nn.Tanh(),
            )

        else:
            raise Exception('Only resNet50 model is implemented.')

        self.segmentRNN_=segmentRNN
        self.contextRNN_=contextRNN

        if opts.semantic_reg:
            self.semantic_branch=nn.Linear(opts.embDim,opts.numClasses)

        def forward(self,x,y1,y2,z1,z2):#we need to check how the input is going to be provided to the model
            #text embedding
            text_emb=self.table([self.contextRNN(y1,y2),self.segmentRNN(z1,z2)],1) #joining on the last dim
            text_emb=self.txt_embedding(text_emb)
            text_emb=norm(text_emb)

            #visual embdding
            visual_emb=self.visionMLP(x)
            visual_emb=visual_emb.view(visual_emb.size(0),-1)
            visual_emb=self.visual_embedding(visual_emb)
            visual_emb=norm(visual_emb)

            if opts.semantic_reg:
                visual_sem=self.semantic_branch(visual_emb)
                text_sem=self.semantic_branch(text_emb)
                #final output
                output=[visual_emb,text_emb,visual_sem,text_sem]
            else:
                #final output
                output=[visual_emb,text_emb]

            return output      
