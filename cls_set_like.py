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
import os
sys.path.append(os.path.realpath('long-range-arena'))

print('Import dataset', flush=True)
from lra_benchmarks.text_classification.input_pipeline import get_tc_datasets


# In[2]:

print('Loading dataset', flush=True)
train_dataset, valid_dataset, test_dataset, encoder = get_tc_datasets(1, 'imdb_reviews', batch_size=BPE_CLS_SETUP['batch_size'], max_length=BPE_CLS_SETUP['max_length'])

# In[7]:


BPE_CLS_SETUP['model_type'] = SetLikeClassifier
BPE_CLS_SETUP['device'] = 'cuda'

def model_postprocess(model):
  for block in model.blocks:
    block.attention_unpack.q = block.attention.q
    block.attention_unpack.k = block.attention.k
        

print('Test instantiation 1', flush=True)
model, criterion, optimizer, schedule_func, scheduler = training_setup(BPE_CLS_SETUP, encoder.vocab_size, model_postprocess)
print(model)

# In[ ]:

for device in range(torch.cuda.device_count()):
  t = torch.cuda.get_device_properties(device).total_memory
  r = torch.cuda.memory_reserved(device) / 1024 ** 2
  a = torch.cuda.memory_allocated(device) / 1024 ** 2

  t = t / 1024 ** 2

  device = torch.cuda.get_device_name(device)

  print(f'Device: {device}, memory reserved: {r}, memory allocated: {a}, memory total: {t}', flush=True)


import torch

for i in range(10):
  test_accuracy = [  ]
  torch.cuda.empty_cache()

  path = f'model_to_test_{i}.b'

  model, criterion, optimizer, schedule_func, scheduler = training_setup(BPE_CLS_SETUP, encoder.vocab_size, model_postprocess)

  checkpoint = train_cls_model(BPE_CLS_SETUP, model, path, train_dataset, valid_dataset, optimizer, criterion, scheduler)
  model.load_state_dict(checkpoint['model_state_dict'])
  
  _, _, acc = cls_test(model, criterion, test_dataset, BPE_CLS_SETUP['device'])
  test_accuracy.append(acc)

  test_accuracy = np.mean(test_accuracy)

  print(f'\nTotal accuracy: {test_accuracy:.4f}')




