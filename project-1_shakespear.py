import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import pandas as pd
import os 
import matplotlib.pyplot as plt
import math
import random
import torch.nn.functional as F

batch_size=64
context_length=256
max_itters=5000
eval_interval=500
eval_itters=200
n_embed=384
lr=3e-4
n_layer=6
n_head=6
dropout=0.2

device ='cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
with open('input.txt','r',encoding='utf-8') as f:
    text=f.read()

chars=sorted(list(set(text)))
vocab_size=len(chars)

s_to_i={ch:i for i,ch in enumerate(chars) }
i_to_s={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[s_to_i[c] for c in s]
decode=lambda l:''.join([i_to_s[i] for i in l])

# print(encode('hi there'))
# print(decode(encode('hi there')))

data=torch.tensor(encode(text),dtype=torch.long) # s-->i

n=int(0.9*len(data))
train_data=data[:n]
val_data=data[n:]

def get_batch(split):
    data=train_data if split=='train' else val_data
    ix=torch.randint(len(data)-context_length,(batch_size,))
    x=torch.stack([data[i:i+context_length] for i in ix])
    y=torch.stack([data[i+1:i+context_length+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y


@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train','val']:
        losses=torch.zeros(eval_itters)

        for k in range(eval_itters):
            x,y=get_batch(split)
            logits,loss=model(x,y)
            losses[k]=loss.item()
        out[split]=losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False)
        self.query=nn.Linear(n_embed,head_size,bias=False)
        self.value=nn.Linear(n_embed,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(context_length,context_length)))
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        B,T,C=x.shape

        k=self.key(x) #(B,T,16)
        q=self.query(x) #(B,T,16)
        weights=q @ k.transpose(-2,-1) *C**-0.5
        weights=weights.masked_fill(self.tril[:T,:T]==0,float('-inf'))
        weights=F.softmax(weights,dim=-1)
        weights=self.dropout(weights)
        v=self.value(x)
        out=weights @ v 
        return out 


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj=nn.Linear(n_embed,n_embed)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)
        out=self.dropout(self.proj(out))

        return out
    

class FeedForward(nn.Module):
    def __init__(self,n_embd):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd), #projection layer that puts it back into the residual path way
            nn .Dropout(dropout)
        )

    def forward(self,x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd,n_head):
        super().__init__()
        head_size=n_embd//n_head
        self.sa_head=MultiHeadAttention(n_head,head_size)
        self.ffw=FeedForward(n_embd)
        self.layerNorm1=nn.LayerNorm(n_embed)
        self.layerNorm2=nn.LayerNorm(n_embed)

    def forward(self,x):
        x=x+self.sa_head(self.layerNorm1(x))
        x=x+self.ffw(self.layerNorm2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embeding_table=nn.Embedding(vocab_size,n_embed)
        self.position_embeding_table=nn.Embedding(context_length,n_embed)
        self.lm_head=nn.Linear(n_embed,vocab_size)
        self.blocks=nn.Sequential(*[Block(n_embed,n_head) for _ in range(n_layer)])
        self.ln_f=nn.LayerNorm(n_embed)

    def forward(self,idx,targets=None):

        idx=idx.to(device)
        B,T=idx.shape
        token_emb=self.token_embeding_table(idx).to(device) #(B,context_lenght or time of sequence, length of embeded vector= vocab size  )
        pos_emb=self.position_embeding_table(torch.arange(T,device=device)) #(T,C)
        x=token_emb+pos_emb # (B,T,C)
        x=self.blocks(x)
        x=self.ln_f(x)
        logits=self.lm_head(x)
        if targets==None:
            loss=None
        else:
            B ,T, C =logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits, targets)
        
        return logits,loss 

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond=idx[:,-context_length:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model=BigramLanguageModel().to(device)
optimizer=torch.optim.AdamW(model.parameters(),lr=lr)

for itter in range(max_itters):
    if itter%eval_interval==0:
        losses=estimate_loss()
        print(f"step {itter}: train loss :{losses['train']:.4f}, val loss :{losses['val']:.4f}")
    xb,yb=get_batch('train')

    logits,loss=model(xb,yb)
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
   


context=torch.zeros((1,1),dtype=torch.long).to(device)
print(decode(model.generate(context,max_new_tokens=500)[0].tolist()))

# open('more.txt','w').write(decode(model.generate(context,max_new_tokens=1000)[0].tolist()))
with open('more.txt', 'w') as file:
    text = decode(model.generate(context, max_new_tokens=10000)[0].tolist())
    file.write(text)


