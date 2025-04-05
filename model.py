import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.module):

    # we will tell the constructor the dimension of the model and the vocabulary size
    def _init_(self, d_model:int, vocab_size: int):
        super()._init_()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    # in the forward the weight of each embeddings layer is multiplied by de sqrt of d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    

    #we will build a matrix of shapee sequence length to d_model
class PostionalEncoding(nn.Module):
    def _init_(self, d_model:int, seq_len:int,dropout:float)-> None:
        super()._init_()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

    #createe a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
    #create a vector of shape(this vector can go from 0 to seq_len-1)
        position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
        div_term= torch.exp(torch.arange(0,d_model,2).float()* (-math.log(10000.0)/d_model))
    #Apply the sin to even positions
        pe[:,0::2]=torch.sin(position *div_term)
        pe[:,1::2]= torch.cos(position *div_term)