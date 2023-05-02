#!/usr/bin/env python
# coding: utf-8

# In[20]:


#get_ipython().system('pwd')


# In[1]:


from lra.layers import *
from lra.models import *
from lra.setups import *
from lra.utils  import *

import sys
sys.path.append('/lnet/troja/work/people/yorsh/lra/long-range-arena')

print('Import dataset', flush=True)
from lra_benchmarks.text_classification.input_pipeline import get_tc_datasets


# In[2]:

print('Loading dataset', flush=True)
train_dataset, valid_dataset, test_dataset, encoder = get_tc_datasets(1, 'imdb_reviews', batch_size=BPE_CLS_SETUP['batch_size'], max_length=BPE_CLS_SETUP['max_length'])

# In[7]:


BPE_CLS_SETUP['model_type'] = LunaClassifier
BPE_CLS_SETUP['device'] = 'cuda'

#def att_factory(hidden_dim, qkv_dim, num_heads, dropout_rate):
#    
#    lka = nn.Sequential(
#        AMGOLU(num_heads, qkv_dim, qkv_dim // num_heads // 4, dropout_rate, nn.Sigmoid(), nn.Identity(), False, LAMBDA=0.0),
#        AMGOLU(num_heads, qkv_dim, qkv_dim // num_heads // 4, dropout_rate, nn.Sigmoid(), nn.Identity(), False, LAMBDA=0.0),
#        AMGOLU(num_heads, qkv_dim, qkv_dim // num_heads // 4, dropout_rate, nn.Sigmoid(), nn.Softplus(), False, LAMBDA=0.0),
        
        #GatedOrthoKernel(num_heads, hidden_dim, dropout_rate, nn.Sigmoid(), nn.Identity(), False, LAMBDA=0.1),
        #GatedOrthoKernel(num_heads, hidden_dim, dropout_rate, nn.Sigmoid(), nn.Identity(), False, LAMBDA=0.1),
        #GatedOrthoKernel(num_heads, hidden_dim, dropout_rate, nn.Sigmoid(), nn.Softplus(), False, LAMBDA=0.1)

        #HeadWiseFF(num_heads, qkv_dim, dropout_rate, nn.Softplus(), use_bias=False, LAMBDA=0.1),   
#    )
#    return LKAAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, lka)
    
    #return FtAttention(hidden_dim, qkv_dim, num_heads, dropout_rate)
    #return SimpleAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, use_lin=True)
    #return SimpleAttention(hidden_dim, qkv_dim, num_heads, dropout_rate, use_lin=False)

print('Test instantiation 1', flush=True)
att_factory=None
model, criterion, optimizer, schedule_func, scheduler = training_setup(BPE_CLS_SETUP, encoder.vocab_size, att_factory)
print(model)

# In[ ]:

for device in range(torch.cuda.device_count()):
  t = torch.cuda.get_device_properties(device).total_memory
  r = torch.cuda.memory_reserved(device)
  a = torch.cuda.memory_allocated(device)

  t = t / 1024 ** 2

  device = torch.cuda.get_device_name(device)

  print(f'Device: {device}, memory reserved: {r}, memory allocated: {a}, memory total: {t}', flush=True)


import torch

test_accuracy = [  ]

torch.cuda.empty_cache()


for i in range(1): ####!!!!!!!!!!!!!!
  path = 'model_to_test_' + str(i) + '.b'

  model, criterion, optimizer, schedule_func, scheduler = training_setup(BPE_CLS_SETUP, encoder.vocab_size, att_factory)

  checkpoint = train_cls_model(BPE_CLS_SETUP, model, path, train_dataset, valid_dataset, optimizer, criterion, scheduler)
  model.load_state_dict(checkpoint['model_state_dict'])
  
  _, _, acc = cls_test(model, criterion, test_dataset, BPE_CLS_SETUP['device'])
  test_accuracy.append(acc)

test_accuracy = np.mean(test_accuracy)

print(f'\nTotal accuracy: {test_accuracy:.4f}')


# In[9]:


model


# In[ ]:




