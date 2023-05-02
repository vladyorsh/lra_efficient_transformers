import torch
import numpy as np
import torch.nn as nn
import math

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

  def forward(self, input):
    batch_size, seq_len = input.shape
    
    embed = self.embedding(input)
    embed = embed + self.pos_embeds
    embed = torch.cat([ self.cls.expand(batch_size, 1, -1), embed ], axis=1)

    return embed

class TAttention(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate):
    super(TAttention, self).__init__()
    self.hidden_dim=hidden_dim
    self.qkv_dim   =qkv_dim
    self.num_heads =num_heads
    
    assert not qkv_dim % num_heads
    
    self.head_dim = qkv_dim // num_heads

    self.q = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)
    self.k = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)
    self.v = nn.Linear(self.hidden_dim, self.qkv_dim, bias=False)

    self.lin = nn.Linear(self.qkv_dim, self.hidden_dim, bias=False)

    self.dropout = nn.Dropout(dropout_rate)

  def split_heads(self, x):
    new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
    x = x.view(* new_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, q, k=None, v=None, losses=[], euclidean=False):
    if k is None:
      k = v = q
    
    q = self.q(q)
    k = self.k(k)
    v = self.v(v)

    
    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    if not euclidean:
      q = torch.mul(q, 1. / torch.sqrt(torch.tensor(self.qkv_dim)))
      qk = torch.matmul(q, k.transpose(-1, -2))
    else:
      qk = torch.cdist(q, k)
    qk = nn.Softmax(dim=-1)(qk)

    def assertion_function(tsr):
      tsr = torch.sum(tsr, axis=-1)
      tsr = tsr - torch.ones_like(tsr)
      return torch.max(torch.abs(tsr)) < 1e-5

    assert assertion_function(qk)

    qk = self.dropout(qk) #Like in TF implementation; could be done before Softmax by random -inf addition

    out = torch.matmul(qk, v)
    out = out.permute(0, 2, 1, 3)

    new_shape = out.shape[:-2] + (self.qkv_dim,)

    out = out.reshape(* new_shape)

    out = self.lin(out)

    return out

class HWLinear(nn.Module):
  def __init__(self, num_heads, input_dim, output_dim, use_bias):
    super(HWLinear, self).__init__()
    
    self.use_bias = use_bias
    if use_bias:
      self.bias   = nn.Parameter(torch.zeros( (1, num_heads, 1, output_dim)))

    self.weight = nn.Parameter(torch.empty( (num_heads, input_dim, output_dim)))

    def he_init(m):
      s =  np.sqrt( 2. / input_dim )
      m.data.normal_(0, s)

    he_init(self.weight)

  def forward(self, x):
    x = torch.matmul(x, self.weight)
    if self.use_bias:
      x += self.bias
    return x


#Layers for kernelized transformers
class HeadWiseFF(nn.Module):
  def __init__(self, num_heads, hidden_dim, dropout_rate, nonlinearity=nn.Identity(), use_bias=False, residual=False, LAMBDA=0.0):
    super(HeadWiseFF, self).__init__()

    head_dim = hidden_dim // num_heads

    self.bias   = nn.Parameter(torch.empty( (1, num_heads, 1, head_dim)))
    self.dropout= nn.Dropout(dropout_rate)
    self.use_bias = use_bias

    #self.weight = nn.Parameter(torch.empty( (num_heads, head_dim, head_dim)))
    #nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    #Orthogonal initialization
    #Workaround with torch.stack, since Torch initializes a tensor as orthgonal by flattening its trailing dims and QR-factorizing the resulting 2d

    self.weight = torch.stack([ nn.init.orthogonal_(torch.empty((head_dim, head_dim))) for _ in range(num_heads) ], dim=0)
    self.weight = nn.Parameter(self.weight)

    bound = 1 / math.sqrt(head_dim)
    nn.init.uniform_(self.bias, -bound, bound)

    self.nonlinearity = nonlinearity
    self.residual= residual

    self.loss = nn.MSELoss()
    self.LAMBDA = LAMBDA

  def forward(self, x):

    x, losses = x

    bs, hd, seq, hdim = x.shape
    y = self.dropout(x)
    y = torch.matmul(y, self.weight) #BS, HD, SEQ, HDIM
    if self.use_bias:
      y += self.bias
    y = self.nonlinearity(y)

    loss = torch.eye(hdim, device=self.weight.device).unsqueeze(0).expand(* self.weight.shape)
    loss = self.loss(torch.matmul(self.weight, self.weight.transpose(-1, -2)), loss)
    loss *= self.LAMBDA

    losses.append(loss)

    if self.residual:
      return x + y, losses
    return y, losses

#LKA aux layers
class AMGOLU(nn.Module):
  def __init__(self, num_heads, hidden_dim, gate_rank, dropout_rate, gate_nonlinearity, kernel_nonlinearity, use_bias=False, LAMBDA = 0.0):
    super(AMGOLU, self).__init__()

    self.head_dim = hidden_dim // num_heads
    self.num_heads= num_heads

    self.orth_weight = HWLinear(num_heads, self.head_dim, self.head_dim, use_bias)
    self.orth_weight.weight = nn.Parameter(torch.stack([ nn.init.orthogonal_(torch.empty((self.head_dim, self.head_dim))) for _ in range(num_heads) ], dim=0))

    self.gate_weight_a = HWLinear(num_heads, self.head_dim, gate_rank, use_bias)
    self.gate_weight_b = HWLinear(num_heads, gate_rank, self.head_dim, use_bias)

    self.kernel_nonlinearity = kernel_nonlinearity
    self.gate_nonlinearity   = gate_nonlinearity

    self.dropout = nn.Dropout(dropout_rate)
    self.LAMBDA = LAMBDA

  def forward(self, x):
    x, losses = x
    x = self.dropout(x)

    forward_info = self.orth_weight(x)
    forward_info = self.kernel_nonlinearity(forward_info)

    gate_info = self.gate_weight_a(x)
    gate_info = self.gate_weight_b(gate_info)
    gate_info = self.gate_nonlinearity(gate_info)

    x = forward_info * gate_info

    loss = torch.eye(self.head_dim, device=self.orth_weight.weight.device).unsqueeze(0).expand(self.num_heads, -1, -1)
    loss = nn.MSELoss()(torch.matmul(self.orth_weight.weight, self.orth_weight.weight.transpose(-1, -2)), loss)
    loss *= self.LAMBDA

    losses.append(loss)

    return x, losses

class GatedOrthoKernel(nn.Module):
  def __init__(self, num_heads, hidden_dim, dropout_rate=0.1, gate_nonlinearity=nn.Sigmoid(), kernel_nonlinearity=nn.Identity(), use_bias=False, LAMBDA = 0.0):
    super(GatedOrthoKernel, self).__init__()

    self.head_dim = hidden_dim // num_heads
    self.num_heads = num_heads

    self.orth_weight = HWLinear(num_heads, self.head_dim, self.head_dim, use_bias)
    self.orth_weight.weight = nn.Parameter(torch.stack([ nn.init.orthogonal_(torch.empty((self.head_dim, self.head_dim))) for _ in range(num_heads) ], dim=0))
    self.gate_weight = HWLinear(num_heads, self.head_dim, self.head_dim, use_bias)

    self.kernel_nonlinearity = kernel_nonlinearity
    self.gate_nonlinearity   = gate_nonlinearity

    self.dropout = nn.Dropout(dropout_rate)
    self.LAMBDA = LAMBDA

  def forward(self, x):
    x, losses = x
    x = self.dropout(x)

    x = self.kernel_nonlinearity(self.orth_weight(x)) * self.gate_nonlinearity(self.gate_weight(x))

    loss = torch.eye(self.head_dim, device=self.orth_weight.weight.device).unsqueeze(0).expand(self.num_heads, -1, -1)
    loss = nn.MSELoss()(torch.matmul(self.orth_weight.weight, self.orth_weight.weight.transpose(-1, -2)), loss)
    loss *= self.LAMBDA

    losses.append(loss)

    return x, losses

class LKAAttention(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, lka):
    super(LKAAttention, self).__init__()
    self.hidden_dim=hidden_dim
    self.num_heads =num_heads

    assert not qkv_dim % num_heads
    
    self.head_dim =qkv_dim // num_heads
    self.qkv_dim = qkv_dim
    
    self.q = nn.Linear(self.hidden_dim, self.qkv_dim)
    self.k = nn.Linear(self.hidden_dim, self.qkv_dim)
    self.v = nn.Linear(self.hidden_dim, self.qkv_dim)

    self.lka = lka
    #self.lka = nn.Sequential(GatedOrthoKernel(self.num_heads, self.hidden_dim, dropout_rate, nn.Sigmoid(), nn.Softplus(), False))

    self.lin = nn.Linear(self.qkv_dim, self.hidden_dim)

    self.dropout = nn.Dropout(dropout_rate)

  def split_heads(self, x):
    new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
    x = x.view(* new_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, q, k=None, v=None, losses=[], ** kwargs):
    if k is None:
      k = v = q
    
    q = self.q(q)
    k = self.k(k)
    v = self.v(v)

    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    #BS x HEADS x SEQ x HEAD_DIM
    
    q, _ = self.lka((q, losses))
    k, _ = self.lka((k, losses)) #Use this for var kernel

    q = q / math.sqrt(self.head_dim)
    k = k / math.sqrt(self.head_dim)

    numerator = torch.matmul(k.unsqueeze(-1), v.unsqueeze(-2))
    numerator = numerator.sum(axis=2)
    numerator = torch.matmul(q, numerator)
    
    denominator = k.sum(axis=2).unsqueeze(-1)
    denominator = q.matmul(denominator)

    out = numerator / denominator
    out = out.permute(0, 2, 1, 3)
    
    #TODO: INSERT DROPOUT
    
    new_shape = out.shape[:-2] + (self.qkv_dim,)
    out = out.reshape(* new_shape)

    out = self.lin(out)

    return out

#SimpleTRON attention
class SimpleAttention(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, use_lin):
    super(SimpleAttention, self).__init__()
    self.hidden_dim=hidden_dim
    self.qkv_dim   =qkv_dim
    self.num_heads =num_heads

    assert not qkv_dim % num_heads
    
    self.head_dim = qkv_dim // num_heads
    
    self.q = nn.Linear(self.hidden_dim, self.qkv_dim)
    self.k = nn.Linear(self.hidden_dim, self.qkv_dim)
    self.v = nn.Linear(self.hidden_dim, self.qkv_dim)

    self.dropout = nn.Dropout(dropout_rate)
    
    self.use_lin = use_lin
    if use_lin:
        assert qkv_dim == hidden_dim
        self.lin = nn.Linear(self.qkv_dim, self.hidden_dim)

  def split_heads(self, x):
    new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
    x = x.view(* new_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, q, k=None, v=None, losses=[], ** kwargs):
    if k is None:
      k = v = q
    
    q = self.q(q)
    k = self.k(k)
    v = self.v(v)

    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v) #BS x HEADS x SEQ x HEAD_DIM

    _, _, seq_len, _ = q.shape

    kv = torch.matmul(k.transpose(-1, -2), v)
    kv *= 1 / math.sqrt(seq_len)
    kv = self.dropout(kv)

    out = torch.matmul(q, kv)
    #out *= 1 / math.sqrt(self.head_dim)
    out = out.permute(0, 2, 1, 3)
    
    new_shape = out.shape[:-2] + (self.qkv_dim,)
    out = out.reshape(* new_shape)
    
    if self.use_lin:
        out = self.lin(out)

    return out

#FNet attention
class FtAttention(nn.Module):
  def __init__(self, *args, **kwargs):
    super(FtAttention, self).__init__()

  def forward(self, q, * args, ** kwargs):
    return torch.fft.fft(torch.fft.fft(q, dim=-1), dim=-2).real

class TBlock(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate):
    super(TBlock, self).__init__()

    self.hidden_dim = hidden_dim
    self.qkv_dim  = qkv_dim
    self.mlp_dim  = mlp_dim

    self.layernorm_input = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
    self.layernorm_inter = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)

    self.attention = TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate)

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim, mlp_dim, bias=False), nn.GELU(), nn.Dropout(dropout_rate),
        nn.Linear(mlp_dim, hidden_dim, bias=False), nn.Dropout(dropout_rate),
    )


  def forward(self, input, losses=[]):
    x = self.layernorm_input(input)
    x = self.attention(x, losses=losses)

    x = input + x

    y = self.layernorm_inter(x)
    x = self.ffn(y) + x

    return x

#Encoder block with an additional skip
class SBlock(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate):
    super(SBlock, self).__init__()

    self.hidden_dim = hidden_dim
    self.qkv_dim  = qkv_dim
    self.mlp_dim  = mlp_dim

    self.layernorm_input = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
    self.layernorm_inter = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)

    self.attention = TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate)

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim, mlp_dim, bias=False), nn.GELU(), nn.Dropout(dropout_rate),
        nn.Linear(mlp_dim, hidden_dim, bias=False), nn.Dropout(dropout_rate),
    )


  def forward(self, input, losses=[]):
    x = self.layernorm_input(input)
    x = self.attention(x, losses=losses)

    x = input + x

    y = self.layernorm_inter(x)
    x = self.ffn(y) + x + input

    return x

class TClassifier(nn.Module):
  def __init__(self, classes, hidden_dim, inter_dim, dropout_rate):
    super(TClassifier, self).__init__()

    self.layernorm = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)
    self.dropout   = nn.Dropout(dropout_rate)

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim, inter_dim, bias=False), nn.GELU(),
    )
    self.output    = nn.Linear(inter_dim, classes, bias=False)

  def forward(self, x):
    x = self.layernorm(x)
    x = x[:, 0, :]
    x = self.dropout(x)

    x = self.ffn(x)
    logits = self.output(x)

    return logits

class DualClassifier(nn.Module):
  def __init__(self, classes, hidden_dim, inter_dim):
    super(DualClassifier, self).__init__()

    self.ffn       = nn.Sequential(
        nn.Linear(hidden_dim * 2, inter_dim, bias=False), nn.ReLU(),
        nn.Linear(inter_dim, inter_dim // 2, bias=False), nn.ReLU(),
    )
    self.output    = nn.Linear(inter_dim // 2, classes, bias=False)

  def forward(self, x):
    emb_1, emb_2 = x
    x = torch.cat([ emb_1, emb_2 ], dim=-1)
    x = x[:, 0, :]
    x = self.ffn(x)
    logits = self.output(x)

    return logits

class LunaBlock(TBlock):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate):
    super(LunaBlock, self).__init__(hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate)
    self.layernorm_mem = nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=False)

    self.attention_unpack = self.attention #TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate)
    #self.attention_unpack.v = self.attention.v
    #self.attention_unpack.lin = self.attention.lin

  def forward(self, input, memory, losses=[]):
    
    packed = self.attention(memory, input, input)
    unpacked=self.attention_unpack(input, packed, packed)

    q = self.layernorm_input(input + unpacked)
    m = self.layernorm_mem(memory + packed)

    y = self.ffn(q)
    q = self.layernorm_inter(q + y)

    return q, m
