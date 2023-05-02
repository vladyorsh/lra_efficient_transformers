import torch.nn as nn
import numpy as np

def transfer_weights(model, original_model, device):
  for name, param in original_model.embeddings.named_parameters():
    param = nn.Parameter(param.cuda(), requires_grad=True)
    
    if name == 'word_embeddings.weight':
      model.embed_layer.embedding.weight = param
    elif name == 'position_embeddings.weight':
      model.embed_layer.pos_embeds = param
    elif name == 'LayerNorm.weight' and model.embed_layer.layernorm.weight.squeeze().shape == param.squeeze().shape:
      model.embed_layer.layernorm.weight = param
    elif name == 'LayerNorm.bias' and model.embed_layer.layernorm.bias.squeeze().shape == param.squeeze().shape:
      model.embed_layer.layernorm.bias = param
    else: print('Unparsable param:', name)
  
  for name, param in original_model.encoder.named_parameters():
    name = name.split('.')[1:]
    
    block_num = int(name[0])

    if block_num >= len(model.blocks): continue
    
    block = model.blocks[block_num]

    error = False

    param = nn.Parameter(param.cuda(), requires_grad=True)

    if name[1] == 'attention':
      if name[2] == 'self':
        layer = None

        if name[3] == 'query':
          layer = block.attention.q
        elif name[3] == 'key':
          layer = block.attention.k
        elif name[3] == 'value':
          layer = block.attention.v
        else: error = True
        
        if name[4] == 'weight':
          layer.weight = param
        elif name[4] == 'bias':
          layer.bias = param
        else: error = True
      
      elif name[2] == 'output':
        if name[3] == 'dense':
          if name[4] == 'weight':
            try:
              block.attention.lin.weight = param
            except:
              error = True
          elif name[4] == 'bias':
            try:
              block.attention.lin.bias = param
            except:
              error = True
          else:
            error = True
        elif name[3] == 'LayerNorm':
          if name[4] == 'weight':
            if block.layernorm_att.weight.squeeze().shape == param.squeeze().shape:
              block.layernorm_att.weight = param
            else: error = True
          elif name[4] == 'bias':
            if block.layernorm_att.bias.squeeze().shape == param.squeeze().shape:
              block.layernorm_att.bias = param
            else: error = True
      else: error = True
    elif name[1] == 'intermediate':
      if name[3] == 'weight':
        block.linear_inter.weight = param
      elif name[3] == 'bias':
        block.linear_inter.bias = param
      else: error = True
    elif name[1] == 'output':
      layer = None
      
      if name[2] == 'dense':
        layer = block.linear_final
      elif name[2] == 'LayerNorm':
        layer = block.layernorm_final
      else: error = True

      if name[3] == 'weight' and layer.weight.squeeze().shape == param.squeeze().shape:
        layer.weight = param
      elif name[3] == 'bias' and layer.bias.squeeze().shape == param.squeeze().shape:
        layer.bias = param
      else: error = True
    else: error = True

    if error: print('Unparsable name', name)
  
  model.pooler.lin.weight = nn.Parameter(original_model.pooler.dense.weight.to(device), requires_grad=True)
  model.pooler.lin.bias   = nn.Parameter(original_model.pooler.dense.bias.to(device), requires_grad=True)
  model = model.to(device)
  return model

def check_weights(model, original_model):
  for name, param in model.named_parameters():
    found = False
    param = param.squeeze()
    for _, o_param  in original_model.named_parameters():
      o_param = o_param.squeeze()

      if param.shape == o_param.shape and ((param - o_param) * (param - o_param)).mean() < 1e-5:
        found = True
        break
    if not found:
      print('Cannot check param:', name)
    
def get_sqrt_schedule(warmup_steps):
  def lr_schedule(step):
    return 1.0 * np.minimum(1.0, step / warmup_steps) / np.sqrt(np.maximum(step, warmup_steps))

  return lr_schedule

def get_stlr_schedule(training_steps, fraction, ratio):
  def lr_schedule(step):
    cut = math.floor(training_steps * fraction)
    p = step / cut if step < cut else 1 - (step-cut)/(cut * (1 / fraction - 1))
    return (1 + p * (ratio - 1)) / ratio
  
  return lr_schedule

def get_warmup_schedule(warmup_steps, ratio):
  def lr_schedule(step):
    p = min(step / warmup_steps, 1.0)
    return p * (1 - ratio) + ratio