import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MiniTransformer(nn.Module):
    """
    The complete tiny transformer.

    Flow: Token Embed + Pos Embed
          → Block 0 (4-head attention + FFN)
          → Block 1 (4-head attention + FFN)
          → LayerNorm → Linear → Logits

    """
    def __init__(self,vocab_size=26, d_model=64, n_heads=4,d_ff=128, n_layers=2, max_seq=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_seq = max_seq

        self.token_emb=nn.Embedding(vocab_size,d_model)
        self.pos_emb=nn.Embedding(max_seq,d_model)

        self.blocks=nn.ModuleList([
            TransformerBlock(d_model,n_heads,d_ff)
            for _ in n_layers
        ])
        self.ln_final=nn.LayerNorm(d_model)
        self.head=nn.Linear(d_model,vocab_size,bias=False)
    
    def forward(self,x,return_attention=False):
        batch,seq_len=x.shape
        positions=torch.arange(seq_len,device=x.device).unsqueeze(0)
        h=self.token_emb(x)+self.pos_emb(positions)

        all_attn=[]
        for block in self.blocks:
            if return_attention:
                h,attn=block(h,return_attention=True)
                all_attn.append(attn)
            else:
                h=block(h)
        
        logits=self.head(self.ln_final(h))
        if return_attention:
            return logits,all_attn
        else:
            return logits
        
# Quick check
test_model = MiniTransformer()
total_params = sum(p.numel() for p in test_model.parameters())
print(f"✓ MiniTransformer defined")
print(f"  Total parameters: {total_params:,}")

dummy = torch.randint(0, 26, (2, 11))
logits, attn = test_model(dummy, return_attention=True)
print(f"  Input shape:  {dummy.shape}")
print(f"  Output shape: {logits.shape}")
print(f"  Attn shapes:  {[a.shape for a in attn]}")        
