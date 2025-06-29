import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter( torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter( torch.rand(d_in, d_out))
        self.W_value = nn.Parameter( torch.rand(d_in, d_out))
        
    def forward(self, x):
        queries = x @ self.W_query
        keys    = x @ self.W_key
        values  = x @ self.W_value
        
        attn_scores= queries@ keys.T
        d_k = keys.shape[-1]
        attn_weights= torch.softmax(attn_scores/ d_k**0.5, dim=-1)
        contexts= attn_weights @ values 
        return contexts
    
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
    
    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores =queries @ keys.T
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        contexts = attn_weights @ values
        return contexts 

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out =d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.context_length = context_length
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        masked = attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        d_k = keys.shape[-1]
        attn_weights = torch.softmax(masked / d_k**0.5, dim=-1)
        context_vec= self.dropout(attn_weights) @ values
        return context_vec

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"
        
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        print(b, num_tokens, self.num_heads,self.head_dim)
        # batch, length, d_out
        queries = self.W_query(x)
        values = self.W_value(x)
        keys= self.W_key(x)
        
        # batch, length, head, dim
        keys = keys.view(b, num_tokens, self.num_heads,self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)  
        
        # batch, head, length, dim
        keys = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values = values.transpose(1,2)
        
        # batch, head, length, dim @ batch, head, dim, length => batch, head, length, length
        attn_scores = queries @ keys.transpose(2,3)
        mask_bool=self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # batch, head, length, length @ batch, head, length, dim => batch, head, length, dim
        # batch, length, head, dim
        context_vec = (attn_weights@values).transpose(1,2)
        
        
        # batch, length, d_out
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return context_vec