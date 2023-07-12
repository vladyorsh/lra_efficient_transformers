import numpy as np
import torch
import tensorflow_datasets as tfds

def num_parameters(model):
  return sum(list(map(
      lambda x: np.prod(x[1].shape), model.named_parameters()
  )))

def get_const_schedule(lr):
  def lr_schedule(step):
    return lr
  return lr_schedule

def get_sqrt_schedule(warmup_steps):
  def lr_schedule(step):
    return 1.0 * np.minimum(1.0, step / warmup_steps) / np.sqrt(np.maximum(step, warmup_steps))

  return lr_schedule
  
#TODO: Seed/shuffling
def torch_generator_wrapper(iterable):
  for item in tfds.as_numpy(iterable):
    yield { key : torch.from_numpy(value) for key, value in item.items() }