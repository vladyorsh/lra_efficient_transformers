import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

from .models import *
from .utils import *

from contextlib import nullcontext

BPE_CLS_SETUP = {
    'model_type' : ClassificationTransformer,
    
    'batch_size' : 4, #32 in total
    'accumulation_steps' : 8,
    'max_length':4000,
    
    'lr' : 0.05,
    'weight_decay' : 0.1,
    'warmup' : 8000,
    'steps' : 20000, #'steps' : 15000,
    'eval_period' : 200,
    'skip_eval' : 0,
    
    'classes' : 2,
    'hidden_dim' : 256,
    'qkv_dim' : 256,
    'mlp_dim': 1024,
    'num_heads' : 4,
    'num_blocks' : 4,
    'output_units' : 1024,
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'device' : 'cuda:0',
    'criterion' : nn.CrossEntropyLoss,
    'optimizer' : optim.AdamW,
    'schedule' : get_sqrt_schedule,
    'mixed_precision' : False,
}

LISTOPS_SETUP = {
    'model_type' : ClassificationTransformer,
    
    'batch_size' : 4,
    'accumulation_steps' : 8,
    'max_length' : 2000,
    
    'lr' : 0.005,
    'weight_decay' : 0.1,
    'warmup': 1000,
    'steps' : 15000,
    'eval_period' : 50,
    'skip_eval' : 0,
    
    'classes' : 10,
    'hidden_dim' : 512,
    'qkv_dim' : 512,
    'num_heads' : 8,
    'num_blocks' : 6,
    'mlp_dim' : 2048,
    'output_units' : 2048,
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'device' : 'cuda:0',
    'criterion' : nn.CrossEntropyLoss,
    'optimizer' : optim.AdamW,
    'schedule' : get_sqrt_schedule,
    'mixed_precision' : True,
}

BPE_MATCH_SETUP = {
    'model_type' : MatchingTransformer,
    
    'batch_size' : 4,
    'accumulation_steps' : 8,
    'max_length' : 4000,
    
    'lr' : 0.05,
    'weight_decay' : 0.1,
    'warmup' : 8000,
    'steps' : 15000,
    'eval_period' : 200,
    'skip_eval' : 0,
    
    'classes' : 2,
    'hidden_dim' : 128,
    'qkv_dim' : 128,
    'mlp_dim' : 512,
    'num_heads' : 4,
    'num_blocks' : 4,
    'output_units' : 512,
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'device' : 'cuda:0',
    'criterion' : nn.CrossEntropyLoss,
    'optimizer' : optim.AdamW,
    'schedule' : get_sqrt_schedule,
    'mixed_precision' : True,
}

#Creates a new instance of a model with given parameters
def model_factory(SETUP, vocab_size, attention_factory):
  model = SETUP['model_type'](
    classes   =SETUP['classes'],
    num_embeddings=vocab_size,
    seq_len   =SETUP['max_length'],
    hidden_dim=SETUP['hidden_dim'],
    qkv_dim   =SETUP['qkv_dim'],
    mlp_dim   =SETUP['mlp_dim'],
    num_heads =SETUP['num_heads'],
    num_blocks=SETUP['num_blocks'],
    output_mlp_units=SETUP['output_units'],
    internal_dropout_rate=SETUP['internal_dropout_rate'],
  ).to(SETUP['device'])

  if attention_factory:
        for block in model.blocks:
            block.attention = attention_factory(SETUP['hidden_dim'], SETUP['qkv_dim'], SETUP['num_heads'], SETUP['internal_dropout_rate']).to(SETUP['device'])
  
  return model

#Returns a model, loss, optimizer and LR scheduler for training
def training_setup(SETUP, vocab_size, attention_factory=None):
    model = model_factory(SETUP, vocab_size, attention_factory)
    criterion = SETUP['criterion']().to(SETUP['device'])
    optimizer = SETUP['optimizer'](model.parameters(), lr=SETUP['lr'], weight_decay=SETUP['weight_decay'])
    schedule_func = SETUP['schedule'](SETUP['warmup'])
    scheduler = LambdaLR(optimizer, schedule_func)
    return model, criterion, optimizer, schedule_func, scheduler

#Training loops for classification, matching and lops models
def train_cls_model(SETUP, model, name, train_dataset, valid_dataset, optimizer, criterion, scheduler):
    
  accumulation_steps=SETUP['accumulation_steps']
  epoch_len = SETUP['eval_period']
  epochs = SETUP['steps'] // epoch_len
  device = SETUP['device']
  skip_eval=SETUP['skip_eval']
  mixed_precision = SETUP['mixed_precision']

  best_acc = 0.0

  bnum = math.ceil(len(train_dataset) / accumulation_steps)
  train_dataset = train_dataset.shuffle(len(train_dataset), reshuffle_each_iteration=True)

  times_repeat = epochs if epoch_len is None else math.ceil(epochs * epoch_len / bnum)
  train_dataset = train_dataset.repeat(times_repeat)
  train_datagen = iter(train_dataset)
  
  if epoch_len is not None:
    bnum = epoch_len
  
  for epoch in range(epochs):  # loop over the dataset multiple times

      #epoch start timestamp
      t = time.time()

      running_loss = 0.0
      running_reg  = 0.0
      running_acc  = 0.0

      running_momentum = 0.99

      epoch_loss = [  ]
      epoch_reg  = [  ]
      epoch_acc  = [  ]

      model.train()

      print(f'Epoch {epoch}')

      process_inputs = lambda x: torch.Tensor(x.numpy()).to(torch.int64)

      scaler = torch.cuda.amp.GradScaler()

      for i in range(bnum):
          # zero the parameter gradients
          optimizer.zero_grad()

          #accumulate gradients for a certain amount of steps
          for k in range(accumulation_steps):
            # get the inputs; data is a list of [inputs, labels]

            try:
              data = next(train_datagen)
            except:
              break
            inputs, labels = data['inputs'], data['targets']
            inputs, labels = process_inputs(inputs), process_inputs(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            with torch.autocast(device_type='cuda', dtype=torch.float16) if mixed_precision else nullcontext():
              outputs, additional_losses = model(inputs)
              loss = criterion(outputs, labels)

            additional_losses = sum(additional_losses) if additional_losses else torch.Tensor([ 0.0 ]).to(device)
            total_loss = (loss + additional_losses) / accumulation_steps

            if mixed_precision: total_loss = scaler.scale(scaled_loss)
            total_loss.backward()

            acc = accuracy(outputs, labels)
            loss = loss.detach()

            running_loss = running_loss * running_momentum + (1 - running_momentum) * loss.item()
            running_loss_unb = running_loss / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            running_acc  = running_acc  * running_momentum + (1 - running_momentum) * acc
            running_acc_unb = running_acc / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            running_reg  = running_reg  * running_momentum + (1 - running_momentum) * additional_losses.item()
            running_reg_unb = running_reg / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            epoch_reg.append(additional_losses.item())

          if mixed_precision:
            scaler.step(optimizer)
            scaler.update()
          else:
            optimizer.step()

          pbar = progress_bar(20, bnum, i + 1)

          print(f'\r{pbar} {i + 1}/{bnum}:', end='', flush=True)
          print(f' - running_loss: {running_loss_unb:.4f} - running_reg: {running_reg_unb:.6f} - running_acc: {running_acc_unb:.4f} - lr: {scheduler.get_last_lr()[0]:.5f}', end='', flush=True)

          scheduler.step()
      
      epoch_loss = np.mean(epoch_loss)
      epoch_acc  = np.mean(epoch_acc)
      epoch_reg  = np.mean(epoch_reg)
      
      print(f' - epoch_loss: {epoch_loss:.4f} - epoch_reg: {epoch_reg:.6f} - epoch_acc: {epoch_acc:.4f}', end='')

      epoch_loss, epoch_acc, epoch_reg = [], [], []

      
      if epoch >= skip_eval:
        model.eval()
        valid_dataset.repeat()

        with torch.no_grad():
          for i, data in enumerate(iter(valid_dataset)):
            inputs, labels = data['inputs'], data['targets']
            inputs, labels = process_inputs(inputs), process_inputs(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16) if mixed_precision else nullcontext():
              outputs, aux_losses = model(inputs)
              loss = criterion(outputs, labels)

            acc = accuracy(outputs, labels)
            aux_losses = sum(aux_losses) if aux_losses else torch.Tensor([ 0.0 ]).to(device)

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            epoch_reg.append(aux_losses.item())

        epoch_loss, epoch_acc, epoch_reg = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_reg)

        if epoch_acc > best_acc:
          best_acc = epoch_acc
          save_model(model, optimizer, name)
      
      else:
        epoch_loss, epoch_acc, epoch_reg = 0.0, 0.0, 0.0

      #epoch computing time
      t = time.time() - t

      print(f' - valid_loss: {epoch_loss:.4f} - valid_reg: {epoch_reg:.6f} - valid_acc: {epoch_acc:.4f} - epoch_time: {t:.4f} s')
 
  checkpoint = torch.load(name)
  return checkpoint

def train_matching_model(SETUP, model, name, train_dataset, valid_dataset, optimizer, criterion, scheduler):

  accumulation_steps=SETUP['accumulation_steps']
  epoch_len = SETUP['eval_period']
  epochs = SETUP['steps'] // epoch_len
  device = SETUP['device']
  skip_eval=SETUP['skip_eval']

  best_acc = 0.0
  train_datagen = iter(train_dataset)
      
  for epoch in range(epochs):  # loop over the dataset multiple times
      
      #epoch start timestamp
      t = time.time()

      running_loss = 0.0
      running_reg  = 0.0
      running_acc  = 0.0

      running_momentum = 0.99

      epoch_loss = [  ]
      epoch_reg  = [  ]
      epoch_acc  = [  ]

      model.train()

      print(f'Epoch {epoch}')

      process_inputs = lambda x: torch.Tensor(x.numpy()).to(torch.int64)

      for i in range(epoch_len):
          # zero the parameter gradients
          optimizer.zero_grad()

          #accumulate gradients for a certain amount of steps
          for k in range(accumulation_steps):
            # get the inputs; data is a list of [inputs, labels]

            try:
              data = next(train_datagen)
            except StopIteration:
              train_datagen = iter(train_dataset)
              data = next(train_datagen)
            except:
              break
            inputs1, inputs2, labels = data['inputs1'], data['inputs2'], data['targets']
            inputs1, inputs2, labels = process_inputs(inputs1), process_inputs(inputs2), process_inputs(labels)
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            # forward + backward + optimize
            outputs, additional_losses = model((inputs1, inputs2))
            loss = criterion(outputs, labels)

            if torch.any(torch.isnan(loss)):
              print(loss)
              return None

            additional_losses = sum(additional_losses) if additional_losses else torch.Tensor([ 0.0 ]).to(device)
            ((loss + additional_losses / 2) / accumulation_steps).backward() #multiply by 1/2 since we have a double input

            acc = accuracy(outputs, labels)

            running_loss = running_loss * running_momentum + (1 - running_momentum) * loss.item()
            running_loss_unb = running_loss / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            running_acc  = running_acc  * running_momentum + (1 - running_momentum) * acc
            running_acc_unb = running_acc / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            running_reg  = running_reg  * running_momentum + (1 - running_momentum) * additional_losses.item()
            running_reg_unb = running_reg / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            epoch_reg.append(additional_losses.item())

          optimizer.step()

          pbar = progress_bar(20, epoch_len, i + 1)

          print(f'\r{pbar} {i + 1}/{epoch_len}:', end='')
          print(f' - running_loss: {running_loss_unb:.4f} - running_reg: {running_reg_unb:.6f} - running_acc: {running_acc_unb:.4f} - lr: {scheduler.get_last_lr()[0]:.5f}', end='')

          scheduler.step()
      
      epoch_loss = np.mean(epoch_loss)
      epoch_acc  = np.mean(epoch_acc)
      epoch_reg  = np.mean(epoch_reg)
      
      print(f' - epoch_loss: {epoch_loss:.4f} - epoch_reg: {epoch_reg:.6f} - epoch_acc: {epoch_acc:.4f}', end='')

      epoch_loss, epoch_acc, epoch_reg = [], [], []

      
      if epoch >= skip_eval:
        model.eval()
        valid_dataset.repeat()
        valid_datagen = iter(valid_dataset)

        with torch.no_grad():
          for i, data in enumerate(valid_datagen):

            inputs1, inputs2, labels = data['inputs1'], data['inputs2'], data['targets']
            inputs1, inputs2, labels = process_inputs(inputs1), process_inputs(inputs2), process_inputs(labels)
            inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

            outputs, aux_losses = model((inputs1, inputs2))
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            aux_losses = sum(aux_losses) if aux_losses else torch.Tensor([ 0.0 ]).to(device)
            aux_losses /= 2 #Doubled input

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            epoch_reg.append(aux_losses.item())

        epoch_loss, epoch_acc, epoch_reg = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_reg)

        if epoch_acc > best_acc:
          best_acc = epoch_acc
          save_model(model, optimizer, name)
      
      else:
        epoch_loss, epoch_acc, epoch_reg = 0.0, 0.0, 0.0

      #epoch computing time
      t = time.time() - t

      print(f' - valid_loss: {epoch_loss:.4f} - valid_reg: {epoch_reg:.6f} - valid_acc: {epoch_acc:.4f} - epoch_time: {t:.4f} s')
 
  checkpoint = torch.load(name)
  return checkpoint

def train_listops_model(SETUP, model, name, train_dataset, valid_dataset, optimizer, criterion, scheduler):
  accumulation_steps=SETUP['accumulation_steps']
  epoch_len = SETUP['eval_period']
  epochs = SETUP['steps'] // epoch_len
  device = SETUP['device']
  skip_eval=SETUP['skip_eval']

  best_acc = 0.0
  train_datagen = iter(train_dataset)
      
  for epoch in range(epochs):  # loop over the dataset multiple times
      
      #epoch start timestamp
      t = time.time()

      running_loss = 0.0
      running_reg  = 0.0
      running_acc  = 0.0

      running_momentum = 0.99

      epoch_loss = [  ]
      epoch_reg  = [  ]
      epoch_acc  = [  ]

      model.train()

      print(f'Epoch {epoch}')

      process_inputs = lambda x: torch.Tensor(x.numpy()).to(torch.int64)

      for i in range(epoch_len):
          # zero the parameter gradients
          optimizer.zero_grad()

          #accumulate gradients for a certain amount of steps
          for k in range(accumulation_steps):
            # get the inputs; data is a list of [inputs, labels]

            try:
              data = next(train_datagen)
            except StopIteration:
              train_datagen = iter(train_dataset)
              data = next(train_datagen)
            except:
              break
            inputs, labels = data['inputs'], data['targets']
            inputs, labels = process_inputs(inputs), process_inputs(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            # forward + backward + optimize
            outputs, additional_losses = model(inputs)
            loss = criterion(outputs, labels)

            if torch.any(torch.isnan(loss)):
              print(loss)
              return None

            additional_losses = sum(additional_losses) if additional_losses else torch.Tensor([ 0.0 ]).to(device)
            ((loss + additional_losses) / accumulation_steps).backward()

            acc = accuracy(outputs, labels)

            running_loss = running_loss * running_momentum + (1 - running_momentum) * loss.item()
            running_loss_unb = running_loss / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            running_acc  = running_acc  * running_momentum + (1 - running_momentum) * acc
            running_acc_unb = running_acc / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            running_reg  = running_reg  * running_momentum + (1 - running_momentum) * additional_losses.item()
            running_reg_unb = running_reg / (1 - running_momentum ** (i * accumulation_steps + k + 1))

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            epoch_reg.append(additional_losses.item())

          optimizer.step()

          pbar = progress_bar(20, epoch_len, i + 1)

          print(f'\r{pbar} {i + 1}/{epoch_len}:', end='')
          print(f' - running_loss: {running_loss_unb:.4f} - running_reg: {running_reg_unb:.6f} - running_acc: {running_acc_unb:.4f} - lr: {scheduler.get_last_lr()[0]:.5f}', end='')

          scheduler.step()
      
      epoch_loss = np.mean(epoch_loss)
      epoch_acc  = np.mean(epoch_acc)
      epoch_reg  = np.mean(epoch_reg)
      
      print(f' - epoch_loss: {epoch_loss:.4f} - epoch_reg: {epoch_reg:.6f} - epoch_acc: {epoch_acc:.4f}', end='')

      epoch_loss, epoch_acc, epoch_reg = [], [], []

      
      if epoch >= skip_eval:
        model.eval()
        valid_dataset.repeat()
        valid_datagen = iter(valid_dataset)

        with torch.no_grad():
          for i, data in enumerate(valid_datagen):

            inputs, labels = data['inputs'], data['targets']
            inputs, labels = process_inputs(inputs), process_inputs(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            outputs, aux_losses = model(inputs)
            loss = criterion(outputs, labels)
            acc = accuracy(outputs, labels)
            aux_losses = sum(aux_losses) if aux_losses else torch.Tensor([ 0.0 ]).to(device)

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            epoch_reg.append(aux_losses.item())

        epoch_loss, epoch_acc, epoch_reg = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_reg)

        if epoch_acc > best_acc:
          best_acc = epoch_acc
          save_model(model, optimizer, name)
      
      else:
        epoch_loss, epoch_acc, epoch_reg = 0.0, 0.0, 0.0

      #epoch computing time
      t = time.time() - t

      print(f' - valid_loss: {epoch_loss:.4f} - valid_reg: {epoch_reg:.6f} - valid_acc: {epoch_acc:.4f} - epoch_time: {t:.4f} s')
 
  checkpoint = torch.load(name)
  return checkpoint

#Test loops for classification + lops and matching models
def cls_test(model, criterion, test_dataset, device='cuda:0'):
  epoch_loss, epoch_acc, epoch_reg = [], [], []

  model.eval()
  test_dataset.repeat()

  process_inputs = lambda x: torch.Tensor(x.numpy()).to(torch.int64)

  t = time.time()

  with torch.no_grad():
    for i, data in enumerate(iter(test_dataset)):
      inputs, labels = data['inputs'], data['targets']
      inputs, labels = process_inputs(inputs), process_inputs(labels)
      inputs, labels = inputs.to(device), labels.to(device)

      outputs, aux_losses = model(inputs)
      loss = criterion(outputs, labels)
      acc = accuracy(outputs, labels)
      aux_losses = sum(aux_losses) if aux_losses else torch.Tensor([ 0.0 ]).cuda()

      epoch_loss.append(loss.item())
      epoch_acc.append(acc)
      epoch_reg.append(aux_losses.item())

  t = time.time() - t

  epoch_loss, epoch_acc, epoch_reg = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_reg)

  print(f' - test_loss: {epoch_loss:.4f} - test_reg: {epoch_reg:.6f} - test_acc: {epoch_acc:.4f} - test_time: {t:.4f} s')
  return epoch_loss, epoch_reg, epoch_acc

def matching_test(model, criterion, test_dataset, device='cuda:0'):
  epoch_loss, epoch_acc, epoch_reg = [], [], []

  model.eval()
  test_dataset.repeat()

  process_inputs = lambda x: torch.Tensor(x.numpy()).to(torch.int64)

  t = time.time()

  with torch.no_grad():
    for i, data in enumerate(iter(test_dataset)):
      inputs1, inputs2, labels = data['inputs1'], data['inputs2'], data['targets']
      inputs1, inputs2, labels = process_inputs(inputs1), process_inputs(inputs2), process_inputs(labels)
      inputs1, inputs2, labels = inputs1.to(device), inputs2.to(device), labels.to(device)

      outputs, aux_losses = model((inputs1, inputs2))
      loss = criterion(outputs, labels)
      acc = accuracy(outputs, labels)
      aux_losses = sum(aux_losses) if aux_losses else torch.Tensor([ 0.0 ]).cuda()
      aux_losses /= 2 #Doubled input

      epoch_loss.append(loss.item())
      epoch_acc.append(acc)
      epoch_reg.append(aux_losses.item())

  t = time.time() - t

  epoch_loss, epoch_acc, epoch_reg = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_reg)

  print(f' - test_loss: {epoch_loss:.4f} - test_reg: {epoch_reg:.6f} - test_acc: {epoch_acc:.4f} - test_time: {t:.4f} s')
  return epoch_loss, epoch_reg, epoch_acc
