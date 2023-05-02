import torch
import numpy as np
import torch.nn as nn
import math

class TEmbedding(nn.Module):
  def __init__(self, tokenizer, hidden_dim, seq_length, dropout_rate, norm_dim):
    super(TEmbedding, self).__init__()
    
    self.num_embeddings = tokenizer.vocab_size
    self.hidden_dim     = hidden_dim
    self.seq_length     = seq_length
    self.padding_idx    = tokenizer.pad_token_id

    self.embedding   = nn.Embedding(self.num_embeddings, self.hidden_dim, self.padding_idx)
    self.pos_embeds  = nn.Parameter(torch.zeros(1, self.seq_length, self.hidden_dim))
    self.layernorm   = nn.LayerNorm(norm_dim, eps=1e-6)
    self.dropout     = nn.Dropout(dropout_rate)

  def forward(self, input, mask):
    batch_size, seq_len = input.shape
    
    embed = self.embedding(input)
    embed = embed + self.pos_embeds
    embed = self.layernorm(embed)
    embed = self.dropout(embed)

    ###!!! Masking
    #embed *= mask

    return embed

class TAttention(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate):
    super(TAttention, self).__init__()
    self.hidden_dim=hidden_dim
    self.qkv_dim   =qkv_dim
    self.num_heads =num_heads
    
    assert not qkv_dim % num_heads
    
    self.head_dim = qkv_dim // num_heads

    self.q = nn.Linear(self.hidden_dim, self.qkv_dim)
    self.k = nn.Linear(self.hidden_dim, self.qkv_dim)
    self.v = nn.Linear(self.hidden_dim, self.qkv_dim)

    self.lin = nn.Linear(self.qkv_dim, self.hidden_dim)

    self.dropout = nn.Dropout(dropout_rate)

  def split_heads(self, x):
    new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
    x = x.view(* new_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, x, mask, losses=[]):
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    #!!! No masking
    #BS x SEQ_LEN x 1 -> BS x SEQ_LEN x SEQ_LEN

    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
    q = torch.mul(q, 1. / torch.sqrt(torch.tensor(self.qkv_dim)))

    qk = torch.matmul(q, k.transpose(-1, -2))

    sm_mask = torch.mul(mask, mask.transpose(-1, -2))
    sm_mask = (1 - sm_mask) * -1e6
    sm_mask = sm_mask.unsqueeze(1)

    qk += sm_mask

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

    #out *= mask

    return out

class LKAAttention(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate, lka):
    super(LKAAttention, self).__init__()
    self.hidden_dim=hidden_dim
    self.qkv_dim   = qkv_dim
    self.num_heads =num_heads

    assert not qkv_dim % num_heads
    
    self.head_dim = qkv_dim // num_heads
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

  def forward(self, x, mask, losses=[]):
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    q *= mask
    k *= mask
    v *= mask

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

class SimpleAttention(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, num_heads, dropout_rate):
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
    self.lin = nn.Linear(self.qkv_dim, self.hidden_dim)

  def split_heads(self, x):
    new_shape = x.shape[:-1] + (self.num_heads, self.head_dim)
    x = x.view(* new_shape)
    return x.permute(0, 2, 1, 3)

  def forward(self, x, mask, losses=[]):
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    #!!!! Masking
    q *= mask
    k *= mask
    v *= mask

    q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v) #BS x HEADS x SEQ x HEAD_DIM

    _, _, seq_len, _ = q.shape

    kv = torch.matmul(k.transpose(-1, -2), v)
    
    norm_constant = mask.sum(dim=(-1, -2))
    norm_constant = 1 / torch.sqrt(norm_constant * self.head_dim)
    norm_constant = norm_constant.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    kv *= norm_constant

    #kv *= 1 / math.sqrt(seq_len)
    kv = self.dropout(kv)

    out = torch.matmul(q, kv)
    #out *= 1 / math.sqrt(self.head_dim)
    out = out.permute(0, 2, 1, 3)
    
    new_shape = out.shape[:-2] + (self.qkv_dim,) #Change to head_dim if usng the linear layer
    out = out.reshape(* new_shape)

    #out = self.lin(out)

    #out *= mask

    return out

class TBlock(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, norm_dim):
    super(TBlock, self).__init__()

    self.hidden_dim = hidden_dim
    self.qkv_dim  = qkv_dim
    self.mlp_dim  = mlp_dim

    self.attention = TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate)
    self.dropout_att   = nn.Dropout(dropout_rate)
    self.layernorm_att = nn.LayerNorm(norm_dim, eps=1e-6)

    self.linear_inter    = nn.Linear(hidden_dim, mlp_dim)
    self.act_inter       = nn.GELU()
    self.linear_final    = nn.Linear(mlp_dim, hidden_dim)
    self.dropout_final   = nn.Dropout(dropout_rate)
    self.layernorm_final = nn.LayerNorm(norm_dim, eps=1e-6)


  def forward(self, input, mask, losses=[]):
    x = self.attention(input, mask, losses)
    #linear layer goes here
    x = self.dropout_att(x)
    x = self.layernorm_att(input + x)

    y = self.linear_inter(x)
    y = self.act_inter(y)
    y = self.linear_final(y)
    y = self.dropout_final(y)
    #y *= mask

    x = self.layernorm_final(y + x)
    #tensor_stats(x)

    return x

#TBlock with an additional skip connection
class SBlock(nn.Module):
  def __init__(self, hidden_dim, qkv_dim, mlp_dim, num_heads, dropout_rate, norm_dim):
    super(SBlock, self).__init__()

    self.hidden_dim = hidden_dim
    self.qkv_dim  = qkv_dim
    self.mlp_dim  = mlp_dim

    self.attention = TAttention(hidden_dim, qkv_dim, num_heads, dropout_rate)
    self.dropout_att   = nn.Dropout(dropout_rate)
    self.layernorm_att = nn.LayerNorm(norm_dim, eps=1e-6)

    self.linear_inter    = nn.Linear(hidden_dim, mlp_dim)
    self.act_inter       = nn.GELU()
    self.linear_final    = nn.Linear(mlp_dim, hidden_dim)
    self.dropout_final   = nn.Dropout(dropout_rate)
    self.layernorm_final = nn.LayerNorm(norm_dim, eps=1e-6)


  def forward(self, input, mask, losses=[]):
    x = self.attention(input, mask, losses)
    #linear layer goes here
    x = self.dropout_att(x)
    x = self.layernorm_att(input + x)

    y = self.linear_inter(x)
    y = self.act_inter(y)
    y = self.linear_final(y)
    y = self.dropout_final(y)
    #y *= mask

    x = self.layernorm_final(y + x + input)
    #tensor_stats(x)

    return x

class TPooler(nn.Module):
  def __init__(self, hidden_dim):
    super(TPooler, self).__init__()
    self.lin = nn.Linear(hidden_dim, hidden_dim)
    self.act = nn.Tanh()

  def forward(self, x):
    x = x[:, 0, :]
    x = self.lin(x)
    x = self.act(x)
    return x


class TClassifier(nn.Module):
  def __init__(self, classes, hidden_dim, dropout_rate):
    super(TClassifier, self).__init__()

    self.dropout   = nn.Dropout(dropout_rate)
    self.output    = nn.Linear(hidden_dim, classes)

  def forward(self, x):
    x = self.dropout(x)
    logits = self.output(x)

    return logits


class Transformer(nn.Module):
  def __init__(self, classes, tokenizer, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0, normalize_len=False):
    super(Transformer, self).__init__()
    
    norm_dim = (hidden_dim,) if not normalize_len else (seq_len, hidden_dim)

    self.embed_layer = TEmbedding(tokenizer, hidden_dim, seq_len, internal_dropout_rate, norm_dim)
    self.blocks      = nn.ModuleList([ SBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, norm_dim) for _ in range(num_blocks) ])
    self.pooler      = TPooler(hidden_dim)
    self.classifier  = TClassifier(classes, hidden_dim, output_dropout_rate)

  def forward(self, pixel_values, mask):
    additional_losses = []
    mask = mask.unsqueeze(-1)

    x = self.embed_layer(pixel_values, mask)
    
    for block in self.blocks:
      x = block(x, mask, additional_losses)
    
    #print('-----------------------------')
    x = self.pooler(x)
    x = self.classifier(x)

    return x, additional_losses