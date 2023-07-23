import numpy as np
import torch
import torchmetrics
import lightning.pytorch as pl
import tensorflow_datasets as tfds
import tqdm

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
  
class Artifact:
    def __init__(self, artifact, name, type, log_every):
        self.artifact = artifact #What to log
        if torch.is_tensor(self.artifact):
            self.artifact = self.artifact.detach().cpu().float()
        self.name = name #How to name
        self.type = type #How to log
        self.log_every = log_every #When to log
  
#TF datasets provided by LRA authors are awkward to use with Torch models
#As an ad-hoc solution dataset batches are being prefetched and then treated as the Dataset samples
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
def wrap_lra_tf_dataset(tf_dataset, verbose=True, num_workers=0):
    return torch.utils.data.DataLoader(TFDatasetWrapper(tf_dataset, verbose), collate_fn=lambda x: x[0], pin_memory=True, num_workers=num_workers)
    
class PBar(pl.callbacks.TQDMProgressBar):
    def init_validation_tqdm(self):
        bar = tqdm.tqdm(            
            disable=True,            
        )
        return bar
        
class LossMetric(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, value):
        if torch.is_tensor(value):
            value = value.detach()
        self.values.append(value)

    def compute(self):
        return torch.mean(torch.Tensor(self.values))

class LunaStopperCallback(pl.callbacks.Callback):
    def __init__(self, key='val_accuracy', threshold_acc=0.51, min_evaluations=10):
        self.threshold_acc  = threshold_acc
        self.min_evaluations= min_evaluations
        self.key = key
        
        self.evaluations = 0
        
    def state_dict(self):
        return { 'threshold_acc' : self.threshold_acc, 'min_evaluations' : self.min_evaluations, 'key' : self.key, 'evaluations' : self.evaluations }
        
    def load_state_dict(self, state_dict):
        self.threshold_acc = state_dict['threshold_acc']
        self.min_evaluations=state_dict['min_evaluations']
        self.key = state_dict['key']
        self.evaluations = state_dict['evaluations']
        
    def on_validation_end(self, trainer, pl_module):
        self.evaluations += 1
        
        logs = trainer.callback_metrics  
        if trainer.fast_dev_run or self.key not in logs.keys():
            return
            
        monitored_val = logs[self.key].squeeze()
        should_stop, reason = False, ''
        
        if self.evaluations > self.min_evaluations and torch.lt(monitored_val, self.key):
            self.should_stop = True
            reason = f'Reached threshold value {self.threshold_acc} after {self.evaluations} evaluations, stopping.'
            
        should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
        trainer.should_stop = trainer.should_stop or should_stop
        
        if reason:
            self._log_info(trainer, reason, False)