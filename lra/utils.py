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
  
class TFDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, tf_dataset, verbose=True):
        self.tf_dataset = tfds.as_numpy(tf_dataset)
        self.samples = [  ]
        for i, sample in enumerate(self.tf_dataset):
          self.samples.append(sample)
          if verbose: print('\rFetching dataset samples: ' + str(i) + '...', end='')
        if verbose: print()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

#DataLoader adds an additional batch dimension, which the original dataset already handles
#However, the shuffling will occur only between batches, not across the whole dataset
def wrap_lra_tf_dataset(tf_dataset, verbose=True):
    return torch.utils.data.DataLoader(TFDatasetWrapper(tf_dataset, verbose), collate_fn=lambda x: [ y for y in x[0] ])