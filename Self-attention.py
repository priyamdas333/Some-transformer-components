import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    
    #Define the parameters of the class for object initialization
    def __init__(self,d_model,d_head):
        super.__init__()
        self.W_Q=nn.Linear(d_model,d_head,bias=False)
        self.W_K=nn.Linear(d_model,d_head,bias=False)
        self.W_V=nn.Linear(d_model,d_head,bias=False)
        self.W_O=nn.Linear(d_head,d_model,bias=False)

    #Define the Methods/Functions over the initialized object
    def forward(self,x, return_attention=False):
        
        Q=self.W_Q(x)
        K=self.W_K(x)
        V=self.W_V(x)
        #All the attention computationsare happening in the attention head
        attention_scores=torch.matmul( Q, K.transpose(-2,-1)/math.sqrt(self.d_head))
        #Use causal mask for autoregressive training
        #Through this, the model can't see the future token and hence after predicting
        #the model prediction loss can be computed and w.r.t weights can be adjusted
        seq_len=x.size()
        mask=torch.triu(torch.ones(seq_len,seq_len,device=x.device),diagonal=1).bool()
        attention_scores=attention_scores.masked_fill(mask,float('-inf'))
        attention_weights=F.softmax(attention_scores)
        attention_output=attention_weights @ V
        #Final output
        O=self.W_O(attention_output)
        if return_attention:
            return attention_weights,O
        else:
            return O
        



