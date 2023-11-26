import torch
import numpy as np
import torch.nn as nn
import math
import einops

from .utils import Artifact

#Ordinary Transformer layers
class TEmbedding(nn.Module):
  def __init__(self, num_embeddings, hidden_dim, seq_length=1024, padding_idx=0):
    super(TEmbedding, self).__init__()
    
    self.num_embeddings = num_embeddings
    self.hidden_dim=hidden_dim
    self.seq_length = seq_length
    self.padding_idx = padding_idx

    self.embedding = nn.Embedding(num_embeddings, hidden_dim, padding_idx)
    self.pos_embeds  = nn.Parameter(torch.zeros(1, self.seq_length, self.hidden_dim))

    self.cls = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
    nn.init.xavier_uniform_(self.cls)

  def forward(self, input, mask=None):
    batch_size, seq_len = input.shape
    
    embed = self.embedding(input)
    embed = embed + self.pos_embeds
    embed = torch.cat([ self.cls.expand(batch_size, 1, -1), embed ], axis=1)
    
    if mask is not None:
        mask = nn.functional.pad(mask, (1, 0), value=1.0)

    return embed, mask

class TAttention(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, affine=False):
    super(TAttention, self).__init__()
    self.hidden_dim=hidden_dim
    self.qkv_dim   =qkv_dim
    self.num_heads =num_heads
    
    assert not qkv_dim % num_heads
    
    self.head_dim = qkv_dim // num_heads
    
    self.q = nn.Linear(self.hidden_dim, self.qkv_dim, bias=affine)
    self.k = nn.Linear(self.hidden_dim, self.qkv_dim, bias=affine)
    self.v = nn.Linear(self.hidden_dim, self.qkv_dim, bias=affine)

    self.lin = nn.Linear(self.qkv_dim, self.hidden_dim, bias=affine)

    self.dropout = nn.Dropout(dropout_rate)

  def split_heads(self, x):
    new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
    x = x.view(* new_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, q, k=None, v=None, q_mask=None, k_mask=None, losses=[]):
    if k is None:
      k = v = q
      k_mask = q_mask
    
    q = self.q(q)
    k = self.k(k)
    v = self.v(v)

    
    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    q = torch.mul(q, 1. / torch.sqrt(torch.tensor(self.head_dim)))
    logits_raw = torch.einsum('bhqd,bhkd->bhqk', q, k)
    logits = logits_raw
    if q_mask is None and k_mask is None:
        ...
    else:
        if q_mask is None:
            mask = einops.rearrange(k_mask, 'b k -> b 1 1 k')
        elif k_mask is None:
            mask = einops.rearrange(q_mask, 'b q -> b 1 q 1')
        else:
            mask = torch.einsum('bq,bk->bqk', q_mask, k_mask).unsqueeze(-3)
        logits = logits + -1e5 * (1 - mask)
    
    att = nn.Softmax(dim=-1)(logits)

    att = self.dropout(att) #Like in TF implementation; could be done before Softmax by random -inf addition

    out = torch.matmul(att, v)
    out = out.permute(0, 2, 1, 3)

    new_shape = out.shape[:-2] + (self.qkv_dim,)

    out = out.reshape(* new_shape)

    out = self.lin(out)

    return out, logits_raw
    
class BAttention(TAttention):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, affine=False, weibull_k=10.0, gamma_beta=1e-4, prior_hidden_size=32, anneal_k=0.00015, anneal_b=6.25, eps=1e-5):
    super(BAttention, self).__init__(hidden_dim, qkv_dim, num_heads, dropout_rate, affine)
    
    self.eps = eps

    #Learnable prior FFN
    prior_lin_1 = nn.Linear(self.head_dim, prior_hidden_size)
    prior_lin_2 = nn.Linear(prior_hidden_size, 1)
    
    prior_lin_1.weight.data.normal_(0, np.sqrt(1 / self.head_dim))
    nn.init.zeros_(prior_lin_1.bias)
    prior_lin_2.weight.data.normal_(0, np.sqrt(1 / prior_hidden_size))
    nn.init.zeros_(prior_lin_2.bias)
    
    self.prior_proj  = nn.Sequential(prior_lin_1, nn.ReLU(), prior_lin_2)

    #Weibull parameters
    self.weibull_k = torch.as_tensor(weibull_k)
    self.gamma_beta= torch.as_tensor(gamma_beta)
    
    #KL scheduling
    self.number_of_calls = 0
    self.anneal_k = anneal_k
    self.anneal_b = anneal_b
    
  def anneal_rate(self):
    return 1/(1+math.exp(-self.anneal_k*self.number_of_calls+self.anneal_b))

  def forward(self, q, k=None, v=None, q_mask=None, k_mask=None, losses=[]):
    if k is None:
      k = v = q
      k_mask = q_mask
    
    q = self.q(q)
    k = self.k(k)
    v = self.v(v)

    
    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    q = torch.mul(q, 1. / torch.sqrt(torch.tensor(self.head_dim)))
    logits_raw = torch.einsum('bhqd,bhkd->bhqk', q, k)
    logits = logits_raw
    if q_mask is None and k_mask is None:
        pass
    else:
        if q_mask is None:
            mask = einops.rearrange(k_mask, 'b k -> b 1 1 k')
        elif k_mask is None:
            mask = einops.rearrange(q_mask, 'b q -> b 1 q 1')
        else:
            mask = torch.einsum('bq,bk->bqk', q_mask, k_mask).unsqueeze(-3)
        logits = logits + -1e5 * (1 - mask)
    
    att = nn.Softmax(dim=-1)(logits)
    logprobs = logits #torch.log(att + self.eps)
    
    if self.training:
        #---Bayesian Attention---
        
        #Building the contextualized prior
        dot_gamma = self.prior_proj(k)
        if k_mask is not None:
            mask = einops.rearrange(k_mask, 'b k -> b 1 k 1')
            dot_gamma + -1e5 * (1 - mask)
        prior_weights = nn.functional.softmax(dot_gamma, dim=-2)
        gamma_alpha   = prior_weights.transpose(-1, -2) * self.gamma_beta
        
        #Computing weights
        noise = torch.rand_like(logprobs) * (1 - 2 * self.eps) + self.eps
        att = nn.functional.softmax(
            logprobs - torch.lgamma(1 + 1.0 / self.weibull_k) + 1.0 / self.weibull_k * torch.log(- torch.log(1.0 - noise + self.eps) + self.eps)
        , dim=-1)
        
        #Computing KL
        #Original code
        lpgamma = logprobs - torch.lgamma(1 + 1.0 / self.weibull_k))
        KL = -(gamma_alpha * (lpgamma - np.euler_gamma * gamma_alpha / self.weibull_k \
                             - self.gamma_beta * torch.exp(logprobs) + \
                             gamma_alpha * torch.log(self.gamma_beta + self.eps) - torch.lgamma(gamma_alpha + self.eps))
                             
        KL = KL.mean() * self.anneal_rate()
        losses.append(KL)
        self.number_of_calls += 1
    
    att = self.dropout(att) #Like in TF implementation; could be done before Softmax by random -inf addition
    
    out = torch.matmul(att, v)
    out = out.permute(0, 2, 1, 3)

    new_shape = out.shape[:-2] + (self.qkv_dim,)

    out = out.reshape(* new_shape)

    out = self.lin(out)

    return out, logits_raw

class TBlock(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine=False, logging_frequency=1000):
    super(TBlock, self).__init__()
    self.logging_frequency = logging_frequency

    self.hidden_dim = hidden_dim
    self.qkv_dim  = qkv_dim
    self.mlp_dim  = mlp_dim

    self.layernorm_input = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=affine)
    self.layernorm_inter = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=affine)

    self.attention = TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, affine)

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim, mlp_dim, bias=affine), nn.GELU(), nn.Dropout(dropout_rate),
        nn.Linear(mlp_dim, hidden_dim, bias=affine), nn.Dropout(dropout_rate),
    )


  def forward(self, input, mask, losses=[], artifacts=[]):
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    x = self.layernorm_input(input)
    x, att = self.attention(x, q_mask=mask, losses=losses)
    
    artifacts.append(
        Artifact(att[0], 'att_logits', 'tensor_stack', self.logging_frequency),
    )

    x = input + x
    
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    y = self.layernorm_inter(x)
    
    x = self.ffn(y) + x

    return x

class TClassifier(nn.Module):
  def __init__(self, classes, hidden_dim, inter_dim, dropout_rate, affine=False):
    super(TClassifier, self).__init__()
    self.classes   = classes

    self.layernorm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=affine)
    self.dropout   = nn.Dropout(dropout_rate)

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim, inter_dim, bias=affine), nn.GELU(),
    )
    self.output    = nn.Linear(inter_dim, classes, bias=affine)

  def forward(self, x, mask=None):
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    x = self.layernorm(x)
    x = x[:, 0, :]
    x = self.dropout(x)

    x = self.ffn(x)
    logits = self.output(x)

    return logits

class DualClassifier(nn.Module):
  def __init__(self, classes, hidden_dim, inter_dim, affine=False):
    super(DualClassifier, self).__init__()
    self.classes   = classes

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim * 2, inter_dim, bias=affine), nn.ReLU(),
        nn.Linear(inter_dim, inter_dim // 2, bias=affine), nn.ReLU(),
    )
    self.output    = nn.Linear(inter_dim // 2, classes, bias=affine)

  def forward(self, x):
    emb_1, emb_2 = x
    x = torch.cat([ emb_1, emb_2 ], dim=-1)
    x = x[:, 0, :]
    x = self.ffn(x)
    logits = self.output(x)

    return logits

class LunaBlock(TBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, shared_att='full'):
    super(LunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency)
    self.layernorm_mem = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=affine)

    if shared_att == 'full':
        self.attention_unpack = self.attention
    else:
        self.attention_unpack = TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate)
        if shared_att is not None:
            if 'q' in shared_att:
                self.attention_unpack.q = self.attention.q
            if 'k' in shared_att:
                self.attention_unpack.k = self.attention.k
            if 'v' in shared_att:
                self.attention_unpack.v = self.attention.v
            if 'o' in shared_att:
                self.attention_unpack.lin = self.attention.lin

  def forward(self, input, memory, mask=None, losses=[], artifacts=[]):
    
    packed, packed_att = self.attention(memory, input, input, k_mask=mask)
    unpacked, unpacked_att = self.attention_unpack(input, packed, packed, q_mask=mask)
    if mask is not None:
        unpacked = unpacked * mask.unsqueeze(-1)
    q = self.layernorm_input(input + unpacked)
    m = self.layernorm_mem(memory + packed)

    y = self.ffn(q)
    if mask is not None:
        y = y * mask.unsqueeze(-1)
    q = self.layernorm_inter(q + y)
    
    artifacts.append( (
    Artifact(packed[0], 'packed', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(unpacked[0], 'unpacked', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(packed_att[0], 'packed_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(unpacked_att[0], 'unpacked_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(memory[0], 'input_memory', ('tensor_slice', 'hist'), self.logging_frequency),
    ) )

    return q, m

class PreLunaBlock(LunaBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, shared_att='full'):
    super(PreLunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency)
    self.layernorm_packed  = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=affine)

  def forward(self, input, memory, mask=None, losses=[], artifacts=[]):
    to_append = ()
  
    x = self.layernorm_input(input)
    m = self.layernorm_mem(memory)
    
    packed, packed_att = self.attention(m, x, x, k_mask=mask)
    to_append = to_append + (Artifact(packed[0], 'packed', ('tensor_slice', 'hist'), self.logging_frequency),)
    
    packed = self.layernorm_packed(packed)
    unpacked, unpacked_att = self.attention_unpack(x, packed, packed, q_mask=mask)
    to_append = to_append + (Artifact(unpacked[0], 'unpacked', ('tensor_slice', 'hist'), self.logging_frequency),)
    
    if mask is not None:
        unpacked = unpacked * mask.unsqueeze(-1)
    q = input + unpacked
    m = memory + packed

    y = self.ffn(self.layernorm_inter(q))
    if mask is not None:
        y = y * mask.unsqueeze(-1)
    q = q + y
    
    to_append = to_append + (
    Artifact(packed_att[0], 'packed_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(unpacked_att[0], 'unpacked_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(memory[0], 'input_memory', ('tensor_slice', 'hist'), self.logging_frequency),
    )
    artifacts.append(to_append)

    return q, m
    
class SelfLunaBlock(PreLunaBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, shared_att='full'):
    super(SelfLunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency)
    
    self.layernorm_packed  = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=affine)
    if shared_att == 'full':
        self.attention_self = self.attention
    else:
        self.attention_self = TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate)
        if shared_att is not None:
            if 'q' in shared_att:
                self.attention_self.q = self.attention.q
            if 'k' in shared_att:
                self.attention_self.k = self.attention.k
            if 'v' in shared_att:
                self.attention_self.v = self.attention.v
            if 'o' in shared_att:
                self.attention_self.lin = self.attention.lin
    #self.layernorm_self = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=affine)

  def forward(self, input, memory, mask=None, losses=[], artifacts=[]):
    to_append = ()
  
    x = self.layernorm_input(input)
    m = self.layernorm_mem(memory)
    
    packed, packed_att = self.attention(m, x, x, k_mask=mask)
    to_append = to_append + (Artifact(packed[0], 'packed', ('tensor_slice', 'hist'), self.logging_frequency),)
    
    packed = self.layernorm_packed(packed)
    unpacked, unpacked_att = self.attention_unpack(x, packed, packed, q_mask=mask)
    to_append = to_append + (Artifact(unpacked[0], 'unpacked', ('tensor_slice', 'hist'), self.logging_frequency),)
    
    self_packed, self_att = self.attention_self(packed, packed, packed)
    to_append = to_append + (Artifact(self_packed[0], 'self_packed', ('tensor_slice', 'hist'), self.logging_frequency),)
    
    if mask is not None:
        unpacked = unpacked * mask.unsqueeze(-1)
    q = input + unpacked
    m = memory + self_packed

    y = self.ffn(self.layernorm_inter(q))
    if mask is not None:
        y = y * mask.unsqueeze(-1)
    q = q + y
    
    to_append = to_append + (
    Artifact(packed_att[0], 'packed_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(unpacked_att[0], 'unpacked_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(self_att[0], 'self_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(memory[0], 'input_memory', ('tensor_slice', 'hist'), self.logging_frequency),
    )
    artifacts.append(to_append)

    return q, m
    
class BLunaBlock(TBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, shared_att='full', weibull_k=10.0, gamma_beta=1e-4, prior_hidden_size=32, anneal_k=0.00015, anneal_b=6.25, eps=1e-5):
    super(BLunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency)
    self.layernorm_mem = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=affine)
    self.attention = BAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, affine, weibull_k, gamma_beta, prior_hidden_size, anneal_k, anneal_b, eps)

    if shared_att == 'full':
        self.attention_unpack = self.attention
    else:
        self.attention_unpack = BAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, affine, weibull_k, gamma_beta, prior_hidden_size, anneal_k, anneal_b, eps)
        if shared_att is not None:
            if 'q' in shared_att:
                self.attention_unpack.q = self.attention.q
            if 'k' in shared_att:
                self.attention_unpack.k = self.attention.k
            if 'v' in shared_att:
                self.attention_unpack.v = self.attention.v
            if 'o' in shared_att:
                self.attention_unpack.lin = self.attention.lin

  def forward(self, input, memory, mask=None, losses=[], artifacts=[]):
    
    packed, packed_att = self.attention(memory, input, input, k_mask=mask)
    unpacked, unpacked_att = self.attention_unpack(input, packed, packed, q_mask=mask)
    if mask is not None:
        unpacked = unpacked * mask.unsqueeze(-1)
    q = self.layernorm_input(input + unpacked)
    m = self.layernorm_mem(memory + packed)

    y = self.ffn(q)
    if mask is not None:
        y = y * mask.unsqueeze(-1)
    q = self.layernorm_inter(q + y)
    
    artifacts.append( (
    Artifact(packed[0], 'packed', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(unpacked[0], 'unpacked', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(packed_att[0], 'packed_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(unpacked_att[0], 'unpacked_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(memory[0], 'input_memory', ('tensor_slice', 'hist'), self.logging_frequency),
    ) )

    return q, m