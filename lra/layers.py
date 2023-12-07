import torch
import numpy as np
import torch.nn as nn
import math
import einops

from .utils import Artifact 

#Copy-paste from
#https://github.com/facebookresearch/mega/blob/main/fairseq/modules/scale_norm.py
class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.affine = elementwise_affine
        if elementwise_affine:
            self.scalar = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('scalar', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.constant_(self.scalar, 1.0)

    def forward(self, x):
        mean_square = torch.mean(torch.square(x), dim=self.dim, keepdim=True)
        if self.scalar is not None:
            x = self.scalar * x

        x = x * torch.rsqrt(mean_square + self.eps)
        return x

    def extra_repr(self) -> str:
        return 'dim={dim}, eps={eps}, affine={affine}'.format(**self.__dict__)

#Ordinary Transformer layers
class TEmbedding(nn.Module):
  def __init__(self, num_embeddings, hidden_dim, seq_length, padding_idx=0, use_cls=True):
    super(TEmbedding, self).__init__()
    
    self.num_embeddings = num_embeddings
    self.hidden_dim=hidden_dim
    self.seq_length = seq_length
    self.padding_idx = padding_idx

    self.embedding = nn.Embedding(num_embeddings, hidden_dim, padding_idx)
    self.pos_embeds  = nn.Parameter(torch.zeros(1, self.seq_length, self.hidden_dim))
    
    #Uncomment if pos embeds are initialized with normal, but this tends to reduce the performance
    #nn.init.normal_(self.embedding.weight, std = 1 / math.sqrt(2))
    #self.embedding.weight.data[padding_idx, :] = 0.0
    #nn.init.normal_(self.pos_embeds, std = 1 / math.sqrt(2)) #Since we use abs embeddings, we keep the output std = 1.0    
    
    self.use_cls = use_cls
    self.cls = None
    if use_cls:
        self.cls = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.xavier_uniform_(self.cls)
        
  def extra_repr(self):
    return 'hidden_dim={hidden_dim}, seq_length={seq_length}, use_cls={use_cls}'.format(**self.__dict__)

  def forward(self, input, mask=None):
    batch_size, seq_len = input.shape
    
    embed = self.embedding(input)
    embed = embed + self.pos_embeds
    if self.cls is not None:
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
    
def get_norm(norm_type, dim, dim_size, eps, affine):
    if norm_type == 'layernorm':
        return nn.LayerNorm(dim_size, eps=eps, elementwise_affine=affine)
    elif norm_type == 'scalenorm':
        return ScaleNorm(dim, eps=eps, elementwise_affine=affine)
    else:
        raise ValueError(f'Norm type {norm_type} is not supported')
    
@torch.jit.script
def reparameterize_noise(logprobs, weibull_k, eps):
    return 1.0 / weibull_k * torch.log(- torch.log(torch.rand_like(logprobs) * (1 - 2 * eps) + eps) + eps)
    
@torch.jit.script
def KL_weibull_gamma(logprobs, gamma_alpha, gamma_beta, lpgamma, weibull_k, eps):
    return -(gamma_alpha * lpgamma - np.euler_gamma * gamma_alpha / weibull_k \
                             - gamma_beta * torch.exp(logprobs) + \
                             gamma_alpha * torch.log(gamma_beta + eps) - torch.lgamma(gamma_alpha + eps)).mean()
    
#Check the correct KLD computation and addition
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
    
    logprobs = logits #torch.log(att + self.eps)
    
    if self.training:
        #---Bayesian Attention---
        
        #Building the contextualized prior
        
        #dot_gamma = self.prior_proj(k)
        #if k_mask is not None:
        #    mask = einops.rearrange(k_mask, 'b k -> b 1 k 1')
        #    dot_gamma + -1e5 * (1 - mask)
        #prior_weights = nn.functional.softmax(dot_gamma, dim=-2)
        #gamma_alpha   = prior_weights.transpose(-1, -2) * self.gamma_beta
        
        #Computing weights
        lpgamma = logprobs - torch.lgamma(1 + 1.0 / self.weibull_k)
        noise = reparameterize_noise(logprobs, self.weibull_k, torch.as_tensor(self.eps))
        att = nn.functional.softmax(
            lpgamma + noise
        , dim=-1)
        
        #Computing KL
        
        #KL = KL_weibull_gamma(logprobs, gamma_alpha, self.gamma_beta, lpgamma, self.weibull_k, torch.as_tensor(self.eps))   
        #KL = KL * self.anneal_rate()
        #self.number_of_calls += 1
        #losses.append(KL)
    else:
        att = nn.Softmax(dim=-1)(logits)
    
    att = self.dropout(att) #Like in TF implementation; could be done before Softmax by random -inf addition
    
    out = torch.matmul(att, v)
    out = out.permute(0, 2, 1, 3)

    new_shape = out.shape[:-2] + (self.qkv_dim,)

    out = out.reshape(* new_shape)

    out = self.lin(out)

    return out, logits_raw

class TBlock(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine=False, logging_frequency=1000, norm_type='layernorm'):
    super(TBlock, self).__init__()
    self.logging_frequency = logging_frequency

    self.hidden_dim = hidden_dim
    self.qkv_dim  = qkv_dim
    self.mlp_dim  = mlp_dim

    self.norm_type  = norm_type
    self.norm_input = get_norm(norm_type, -1, self.hidden_dim, 1e-6, affine)
    self.norm_inter = get_norm(norm_type, -1, self.hidden_dim, 1e-6, affine)

    self.attention = TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, affine)

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim, mlp_dim, bias=affine), nn.GELU(), nn.Dropout(dropout_rate),
        nn.Linear(mlp_dim, hidden_dim, bias=affine), nn.Dropout(dropout_rate),
    )

  def forward(self, input, mask, losses=[], artifacts=[]):
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    x = self.norm_input(input)
    x, att = self.attention(x, q_mask=mask, losses=losses)
    
    artifacts.append(
        Artifact(att[0], 'att_logits', 'tensor_stack', self.logging_frequency),
    )

    x = input + x
    
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    y = self.norm_inter(x)
    
    x = self.ffn(y) + x

    return x

class TClassifier(nn.Module):
  def __init__(self, classes, hidden_dim, inter_dim, dropout_rate, affine=False, use_cls=True, norm_type='layernorm'):
    super(TClassifier, self).__init__()
    self.classes   = classes
    self.use_cls   = use_cls
    
    self.norm = get_norm(norm_type, -1, hidden_dim, 1e-6, affine)
    self.dropout   = nn.Dropout(dropout_rate)

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim, inter_dim, bias=affine), nn.GELU(),
    )
    self.output    = nn.Linear(inter_dim, classes, bias=affine)

  def forward(self, x, mask=None):
    if mask is not None:
        x = x * mask.unsqueeze(-1)
    x = self.norm(x)
    if self.use_cls:
        x = x[..., 0, :]
    else:
        x = x.mean(dim=-2)
    x = self.dropout(x)

    x = self.ffn(x)
    logits = self.output(x)

    return logits

class DualClassifier(nn.Module):
  def __init__(self, classes, hidden_dim, inter_dim, affine, use_cls):
    super(DualClassifier, self).__init__()
    self.classes   = classes
    self.use_cls   = use_cls

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim * 2, inter_dim, bias=affine), nn.GELU(),
        nn.Linear(inter_dim, inter_dim // 2, bias=affine), nn.GELU(),
    )
    self.output    = nn.Linear(inter_dim // 2, classes, bias=affine)

  def forward(self, x):
    emb_1, emb_2 = x
    x = torch.cat([ emb_1, emb_2 ], dim=-1)
    if self.use_cls:
        x = x[..., 0, :]
    else:
        x = x.mean(dim=-2)
    x = self.ffn(x)
    logits = self.output(x)

    return logits

class LunaBlock(TBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, norm_type='layernorm', shared_att='full'):
    super(LunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency, norm_type)
    
    self.norm_mem = get_norm(norm_type, -1, self.hidden_dim, 1e-6, affine)

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
    q = self.norm_input(input + unpacked)
    m = self.norm_mem(memory + packed)

    y = self.ffn(q)
    if mask is not None:
        y = y * mask.unsqueeze(-1)
    q = self.norm_inter(q + y)
    
    artifacts.append( (
    Artifact(packed[0], 'packed', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(unpacked[0], 'unpacked', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(packed_att[0], 'packed_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(unpacked_att[0], 'unpacked_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(memory[0], 'input_memory', ('tensor_slice', 'hist'), self.logging_frequency),
    ) )

    return q, m

class ConvAttention(TAttention):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, affine=False, kernel=(4, 1), stride=(1, 1), padding='zeros', pool=False, temperature='unit'):
    super(ConvAttention, self).__init__(hidden_dim, qkv_dim, num_heads, dropout_rate, affine)
    if not pool:
        self.kernel = nn.Conv2d(num_heads, num_heads, kernel, stride, groups=num_heads, bias=False, padding=padding)
    else:
        self.kernel = nn.MaxPool2d(kernel, stride, padding=0)
        
    if temperature == 'unit':
        self.log_temp = nn.Parameter(torch.zeros(1,), requires_grad=False)
    elif temperature == 'sqrt':
        self.log_temp = nn.Parameter(torch.ones(1,) * np.log(self.head_dim), requires_grad=False)
    elif temperature == 'learn':
        self.log_temp = nn.Parameter(torch.zeros(1,), requires_grad=True)
    else:
        raise ValueError(f'Temperature option {temperature} is not supported for unpack attention')

  def forward(self, q, k=None, v=None, q_mask=None, k_mask=None, losses=[]):
    if k is None:
      k = v = q
      k_mask = q_mask
    
    q = self.q(q)
    k = self.k(k)
    v = self.v(v)

    
    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    q = torch.mul(q, 1. / torch.exp(self.log_temp))
    
    o_k, o_v = k, v
    k, v = self.kernel(k), self.kernel(v)
    shape_change = (o_k.shape != k.shape) or (o_v.shape != v.shape)
    
    logits_raw = torch.einsum('bhqd,bhkd->bhqk', q, k)
    logits = logits_raw
    if (q_mask is None and k_mask is None) or shape_change:
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
    
class SimplifiedConvAttention(ConvAttention):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, affine=False, kernel=(4, 1), stride=(1, 1), padding='zeros', pool=False, temperature='unit'):
    super(SimplifiedConvAttention, self).__init__(hidden_dim, qkv_dim, num_heads, dropout_rate, affine, kernel, stride, padding, pool, temperature)
    del self.v
        
  def forward(self, q, k=None, v=None, q_mask=None, k_mask=None, losses=[]):
    if k is None:
      k = v = q
      k_mask = q_mask
    
    q = self.q(q)
    k = self.k(k)
    v = v

    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    q = torch.mul(q, 1. / torch.exp(self.log_temp))
    
    o_k, o_v = k, v
    if k_mask is not None:
        mask = einops.rearrange(k_mask, 'b k -> b 1 k 1')
        k, v = k * mask, v * mask
    k, v = self.kernel(k), self.kernel(v)
    shape_change = (o_k.shape != k.shape) or (o_v.shape != v.shape)
    
    logits_raw = torch.einsum('bhqd,bhkd->bhqk', q, k)
    logits = logits_raw
    if (q_mask is None and k_mask is None) or shape_change:
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
    
class UnpackAttention(TAttention):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, affine=False, temperature='unit'):
    super(UnpackAttention, self).__init__(hidden_dim, qkv_dim, num_heads, dropout_rate, affine)
    if temperature == 'unit':
        self.log_temp = nn.Parameter(torch.zeros(1,), requires_grad=False)
    elif temperature == 'sqrt':
        self.log_temp = nn.Parameter(torch.ones(1,) * np.log(self.head_dim), requires_grad=False)
    elif temperature == 'learn':
        self.log_temp = nn.Parameter(torch.zeros(1,), requires_grad=True)
    else:
        raise ValueError(f'Temperature option {temperature} is not supported for unpack attention')
  
  def forward(self, q, k=None, v=None, q_mask=None, k_mask=None, losses=[]):
    if k is None:
      k = v = q
      k_mask = q_mask
    
    q = self.q(q)
    k = self.k(k)
    v = self.v(v)

    
    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    q = torch.mul(q, 1. / torch.exp(self.log_temp))
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

class SimplifiedUnpackAttention(UnpackAttention):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, affine=False, temperature='unit'):
    super(SimplifiedUnpackAttention, self).__init__(hidden_dim, qkv_dim, num_heads, dropout_rate, affine, temperature)
    del self.v
  
  def forward(self, q, k=None, v=None, q_mask=None, k_mask=None, losses=[]):
    if k is None:
      k = v = q
      k_mask = q_mask
    
    q = self.q(q)
    k = self.k(k)
    v = v
    
    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    q = torch.mul(q, 1. / torch.exp(self.log_temp))
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
    
class ConvLunaBlock(LunaBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, norm_type='layernorm', shared_att='full', kernel=4, stride=1, padding=None, pool=False, temperature_pack='unit', temperature_unpack='unit'):
    super(ConvLunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency, norm_type, shared_att)
    if padding is None:
        padding = 'same' if stride == 1 else 0
            
    self.attention = ConvAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, affine, (kernel, 1), (stride, 1), padding, pool, temperature_pack)
      
    if shared_att == 'full':
      shared_att == 'qkvo'
        
    self.attention_unpack = UnpackAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, temperature_unpack)
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
    q = self.norm_input(input + unpacked)
    m = self.norm_mem(memory + packed)

    y = self.ffn(q)
    if mask is not None:
        y = y * mask.unsqueeze(-1)
    q = self.norm_inter(q + y)
    
    artifacts.append( (
    Artifact(packed[0], 'packed', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(unpacked[0], 'unpacked', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(packed_att[0], 'packed_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(unpacked_att[0], 'unpacked_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(memory[0], 'input_memory', ('tensor_slice', 'hist'), self.logging_frequency),
    ) )

    return q, m
    
class SimplifiedConvLunaBlock(ConvLunaBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, norm_type='layernorm', kernel=4, stride=1, padding=None, pool=False, temperature_pack='unit', temperature_unpack='unit'):
    super(SimplifiedConvLunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency, norm_type, 'qk', kernel, stride, padding, pool, temperature_pack, temperature_unpack)
    if padding is None:
        padding = 'same' if stride == 1 else 0
            
    self.attention = SimplifiedConvAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, affine, (kernel, 1), (stride, 1), padding, pool, temperature_pack)
    self.attention_unpack = SimplifiedUnpackAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, temperature_unpack)
    
    self.attention_unpack.q = self.attention.q
    self.attention_unpack.k = self.attention.k
    self.attention_unpack.lin = self.attention.lin
    
class BLunaBlock(TBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, norm_type='layernorm', shared_att='full', weibull_k=10.0, gamma_beta=1e-4, prior_hidden_size=32, anneal_k=0.00015, anneal_b=6.25, eps=1e-5):
    super(BLunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency, norm_type, shared_att)
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
    
    packed, packed_att = self.attention(memory, input, input, k_mask=mask, losses=losses)
    unpacked, unpacked_att = self.attention_unpack(input, packed, packed, q_mask=mask, losses=losses)
    if mask is not None:
        unpacked = unpacked * mask.unsqueeze(-1)
    q = self.norm_input(input + unpacked)
    m = self.norm_mem(memory + packed)

    y = self.ffn(q)
    if mask is not None:
        y = y * mask.unsqueeze(-1)
    q = self.norm_inter(q + y)
    
    artifacts.append( (
    Artifact(packed[0], 'packed', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(unpacked[0], 'unpacked', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(packed_att[0], 'packed_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(unpacked_att[0], 'unpacked_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(memory[0], 'input_memory', ('tensor_slice', 'hist'), self.logging_frequency),
    ) )

    return q, m
    
class vMFLunaBlock(ConvLunaBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency=1000, norm_type='layernorm', shared_att='full', kernel=4, stride=1, padding=None, pool=False, vmf_k=10.0, mem_size=256, anneal_k=0.00015, anneal_b=6.25, eps=1e-5):
    super(vMFLunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, affine, logging_frequency, norm_type, shared_att, kernel, stride, padding, pool)
    
    self.rec_network = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim)
    )
    [ nn.init.zeros_(b) for n, b in self.rec_network.named_parameters() if 'bias' in n ]
    
    self.vmf_k = nn.Parameter(torch.as_tensor(vmf_k), requires_grad=False)
    self.prior_mu = nn.Parameter(torch.randn(1, 1, mem_size, hidden_dim), requires_grad=True)
    nn.init.normal_(self.prior_mu)
    self.prior_mu.data = self.prior_mu.data / torch.norm(self.prior_mu, dim=-1, keepdim=True)
    
    self.number_of_calls = 0
    self.anneal_k = anneal_k
    self.anneal_b = anneal_b
    
  def anneal_rate(self):
    return 1/(1+math.exp(-self.anneal_k*self.number_of_calls+self.anneal_b))
    
  def sample(self, recognized, rejection_sampling_factor=10):
    v = torch.randn_like(recognized, device=self.vmf_k.device)[..., :-1]
    v = v / torch.norm(v, dim=-1, keepdim=True)
    d = recognized.shape[-1]
    
    kmr = torch.sqrt( 4 * self.vmf_k**2 + (d - 1)**2 )
    bb = (kmr - 2 * self.vmf_k) / (d-1)
    aa = (kmr + 2 * self.vmf_k + d - 1) / 4
    dd = (4 * aa * bb)/(1 + bb) - (d - 1)*np.log(d-1)
    beta = torch.distributions.Beta(torch.tensor(0.5 * (d - 1)), torch.tensor(0.5 * (d - 1)) )
    uniform = torch.distributions.Uniform(0.0, 1.0)
    
    N = np.prod(v.shape[:-1])

    v0 = torch.tensor([]).to(self.vmf_k)
    while len(v0) < N:
        eps = beta.sample([1, rejection_sampling_factor*(N-len(v0))]).squeeze().to(self.vmf_k)
        uns = uniform.sample([1, rejection_sampling_factor*(N-len(v0))]).squeeze().to(self.vmf_k)
        w0 = (1 - (1+bb)*eps) / (1 - (1-bb)*eps)
        t0 = (2*aa*bb) / (1 - (1-bb)*eps)
        det = (d-1)*t0.log() - t0 + dd - uns.log()
        v0 = torch.cat([v0, torch.tensor(w0[det>=0]).to(self.vmf_k)])
        if len(v0) > N:
            v0 = v0[:N]
            break
    v0 = v0.reshape(v.shape[:-1] + (1,))
    
    # Step-3: Form x = [v0; sqrt(1-v0^2)*v]
    samples = torch.cat([v0, (1-v0**2).sqrt()*v], dim=-1)

    # Setup-4: Householder transformation
    e1mu = torch.zeros_like(recognized);  e1mu[..., 0] = 1.0
    e1mu = e1mu - recognized
    e1mu = e1mu / torch.norm(e1mu, dim=-1, keepdim=True)
    U = torch.eye(d, device=self.vmf_k.device)[None, None, ...] - 2 * torch.einsum('...md,...mt->...mdt', e1mu, e1mu)
    samples = torch.einsum('...mtd,...md->...mt',U,samples)
    
    return samples
    
  def prior_unit_constraint(self):
    return torch.square(torch.norm(self.prior_mu, dim=-1) - 1.0)
    
  def kld(self, sample):
    return (1 - nn.functional.cosine_similarity(sample, self.prior_mu, dim=-1) + self.prior_unit_constraint()).mean()

  def forward(self, input, memory, mask=None, losses=[], artifacts=[]):
    
    packed, packed_att = self.attention(memory, input, input, k_mask=mask)
   
    #Recognize the direction vectors
    recognized = self.rec_network(packed)
    recognized = recognized / recognized.sum(dim=-1, keepdim=True)
    
    #Sample from variational distributions
    sample = self.sample(recognized)
    
    #Compute KLD
    kld = self.kld(sample) * self.anneal_rate()
    self.number_of_calls += 1
    losses.append(kld)
    
    #Use sampled vectors as a memory
    unpacked, unpacked_att = self.attention_unpack(input, sample, sample, q_mask=mask)
    if mask is not None:
        unpacked = unpacked * mask.unsqueeze(-1)
    q = self.norm_input(input + unpacked)
    m = self.norm_mem(memory + sample)

    y = self.ffn(q)
    if mask is not None:
        y = y * mask.unsqueeze(-1)
    q = self.norm_inter(q + y)
    
    artifacts.append( (
    Artifact(packed[0], 'packed', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(unpacked[0], 'unpacked', ('tensor_slice', 'hist'), self.logging_frequency),
    Artifact(packed_att[0], 'packed_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(unpacked_att[0], 'unpacked_att_logits', 'tensor_stack', self.logging_frequency),
    Artifact(memory[0], 'input_memory', ('tensor_slice', 'hist'), self.logging_frequency),
    ) )

    return q, m