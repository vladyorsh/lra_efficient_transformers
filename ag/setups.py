import time

from .models import *
from .utils import transfer_weights, check_weights, get_sqrt_schedule
from lra.utils import accuracy, save_model, progress_bar

import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from transformers import AutoTokenizer, BertModel
import datasets

AG_SETUP = {
    'model_type' : AGTransformer,
    'bert_config' : 'bert-base-cased',
    'seed' : 42,
    'device' : 'cuda:1',
    
    'classes' : 4,
    'batch_size' : 4,
    'accumulation_steps' : 8,
    'max_length': 512,
    
    'hidden_dim' : 768,
    'qkv_dim' : 768,
    'mlp_dim' : 3072,
    'num_heads': 12,
    'num_blocks' : 12,
    'output_mlp_units' : 3072,
    'internal_dropout_rate' : 0.2,
    'output_dropout_rate' : 0.0,
    
    'coarse_lr' : 5e-5,
    'coarse_wd' : 1e-4,
    'coarse_ratio' : 32,
    'coarse_periods' : 20,
    'coarse_eval_period' : 525,
    'coarse_criterion' : nn.CrossEntropyLoss,
    'coarse_optimizer' : optim.AdamW,
    'coarse_schedule' : lambda x: 1.0,
    'coarse_skip_eval' : 0,
    
    'fine_lr' : 5e-6,
    'fine_wd' : 1e-4,
    'fine_warmup' : 525,
    'fine_ratio' : 32,
    'fine_periods': 20,
    'fine_eval_period' : 525,
    'fine_lr_per_layer_decay' : 0.9,
    'fine_criterion' : nn.CrossEntropyLoss,
    'fine_optimizer' : optim.AdamW,
    'fine_schedule' : get_sqrt_schedule,
    'fine_skip_eval' : 0,
}

def get_data_tokenizer_bert(SETUP):
    tokenizer = AutoTokenizer.from_pretrained(SETUP['bert_config'])
    b_model   = BertModel.from_pretrained(SETUP['bert_config'])
    
    def mapping_function(batch):
      items = []
      attention_masks = []

      for item in batch['input_ids']:
        if len(item) > 512:
          item = item[:128] + item[-384:]
        elif len(item) < 512:
          item = item + [ tokenizer.pad_token_id ] * (512 - len(item))
        items.append(item)

      for mask in batch['attention_mask']:
        if len(mask) > 512:
          mask = mask[:128] + mask[-384:]
        elif len(mask) < 512:
          mask = mask + [ 0 ] * (512 - len(mask))

        attention_masks.append(mask)

      batch['input_ids'] = items
      batch['attention_mask'] = attention_masks

      return batch
    
    ag_train, ag_valid, ag_test = datasets.load_dataset('ag_news', split=['train[:70%]', 'train[-30%:]', 'test'])
    ag_train, ag_valid, ag_test = [ ds.map(lambda e: tokenizer(e['text'], truncation=False, padding='longest'), batched=True).map(mapping_function, batched=True) for ds in [ ag_train, ag_valid, ag_test ] ]
    [ ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label']) for ds in [ ag_train, ag_valid, ag_test ] ]
    ag_train, ag_valid, ag_test = [ ds.shuffle(seed=42) for ds in [ ag_train, ag_valid, ag_test ] ]
    return ag_train, ag_valid, ag_test, tokenizer, b_model

def model_factory(SETUP, tokenizer, attention_factory):
  model = SETUP['model_type'](
    classes   =SETUP['classes'],
    tokenizer=tokenizer,
    seq_len   =SETUP['max_length'],
    hidden_dim=SETUP['hidden_dim'],
    qkv_dim   =SETUP['qkv_dim'],
    mlp_dim   =SETUP['mlp_dim'],
    num_heads =SETUP['num_heads'],
    num_blocks=SETUP['num_blocks'],
    output_mlp_units=SETUP['output_mlp_units'],
    internal_dropout_rate=SETUP['internal_dropout_rate'],
  ).to(SETUP['device'])

  if attention_factory:
        for block in model.blocks:
            block.attention = attention_factory(SETUP['hidden_dim'], SETUP['qkv_dim'], SETUP['num_heads'], SETUP['internal_dropout_rate']).to(SETUP['device'])
  return model
            

def coarse_training_setup(SETUP, tokenizer, weight_donor=None, attention_factory=None):
    model = model_factory(SETUP, tokenizer, attention_factory)
    if weight_donor is not None:
        weight_donor = weight_donor.to(SETUP['device'])
        model = transfer_weights(model, weight_donor, SETUP['device'])
        check_weights(model, weight_donor)
    
    optim_array = [ { "params" : model.classifier.parameters(), 'lr' : SETUP['coarse_lr'] } ]
    optim_array.append({ "params" : model.pooler.parameters(), 'lr' : SETUP['coarse_lr'] })
  
    for i in range(len(model.blocks)):
        index = len(model.blocks) - 1 - i    
        optim_array.append({'lr' : SETUP['coarse_lr'], 'params': model.blocks[index].attention.parameters()})
    
    criterion = SETUP['coarse_criterion']().to(SETUP['device'])
    optimizer = SETUP['coarse_optimizer'](optim_array, lr=SETUP['coarse_lr'], weight_decay=SETUP['coarse_wd'])
    schedule_func = SETUP['coarse_schedule']
    scheduler = LambdaLR(optimizer, schedule_func)
    return model, criterion, optimizer, schedule_func, scheduler

def fine_training_setup(SETUP, model):
    lr = SETUP['fine_lr']
    
    #Per-layer LR setup
    optim_array = [ { "params" : model.classifier.parameters(), 'lr' : lr } ]
    optim_array.append({ "params" : model.pooler.parameters(), 'lr' : lr })
    lr *= SETUP['fine_lr_per_layer_decay']

    for i in range(len(model.blocks)):
        index = len(model.blocks) - 1 - i
        block = model.blocks[index]
        optim_array.append({
            'params' : block.parameters(),
            'lr' : lr
        })

        lr *= SETUP['fine_lr_per_layer_decay']
    optim_array.append({ 'params' : model.embed_layer.parameters(), 'lr' : lr })
    #####
    criterion = SETUP['fine_criterion']().to(SETUP['device'])
    optimizer = SETUP['fine_optimizer'](optim_array, lr=SETUP['fine_lr'], weight_decay=SETUP['fine_wd'])
    schedule_func = SETUP['fine_schedule'](SETUP['fine_warmup'])
    scheduler = LambdaLR(optimizer, schedule_func)
    return model, criterion, optimizer, schedule_func, scheduler
    
    
def train_ag_model(SETUP, model, name, train_dataset, valid_dataset, optimizer, criterion, scheduler, epochs, epoch_len=None, skip_eval=0):
  device = SETUP['device']
  ACCUMULATION_STEPS = SETUP['accumulation_steps']
  BATCH_SIZE = SETUP['batch_size']

  best_acc = 0.0

  bnum = math.ceil(len(train_dataset) / ACCUMULATION_STEPS / BATCH_SIZE)
  
  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

  train_datagen = iter(train_loader)
        
  if epoch_len is not None:
    bnum = epoch_len
  
  for epoch in range(epochs):  # loop over the dataset multiple times

      #epoch start timestamp
      t = time.time()

      train_datagen = iter(train_loader)

      running_loss = 0.0
      running_reg  = 0.0
      running_acc  = 0.0

      running_momentum = 0.999

      epoch_loss = [  ]
      epoch_reg  = [  ]
      epoch_acc  = [  ]

      model.train()

      print(f'Epoch {epoch}')

      for i in range(bnum):
          # zero the parameter gradients
          optimizer.zero_grad()

          # indicates whether the datagen is empty and it's needed to create new
          terminate_epoch = False

          #print(model.blocks[0].attention.q.weight)

          #accumulate gradients for a certain amount of steps
          for k in range(ACCUMULATION_STEPS):
            # get the inputs; data is a list of [inputs, labels]

            try:
              data = next(train_datagen)
            except:
              terminate_epoch = True
              break
            inputs, labels, mask = data['input_ids'], data['label'], data['attention_mask']
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)

            # forward + backward + optimize
            outputs, additional_losses = model(inputs, mask)
            loss = criterion(outputs, labels)

            additional_losses = sum(additional_losses) if additional_losses else torch.Tensor([ 0.0 ]).to(device)
            ((loss + additional_losses) / ACCUMULATION_STEPS).backward()

            acc = accuracy(outputs, labels)

            running_loss = running_loss * running_momentum + (1 - running_momentum) * loss.item()
            running_loss_unb = running_loss / (1 - running_momentum ** (i * ACCUMULATION_STEPS + k + 1))

            running_acc  = running_acc  * running_momentum + (1 - running_momentum) * acc
            running_acc_unb = running_acc / (1 - running_momentum ** (i * ACCUMULATION_STEPS + k + 1))

            running_reg  = running_reg  * running_momentum + (1 - running_momentum) * additional_losses.item()
            running_reg_unb = running_reg / (1 - running_momentum ** (i * ACCUMULATION_STEPS + k + 1))

            epoch_loss.append(loss.item())
            epoch_acc.append(acc)
            epoch_reg.append(additional_losses.item())

          
          #print(model.blocks[0].attention.q.weight.grad)

          optimizer.step()

          pbar = progress_bar(20, bnum, i + 1)

          if terminate_epoch: break

          print(f'\r{pbar} {i + 1}/{bnum}:', end='')
          print(f' - running_loss: {running_loss_unb:.4f} - running_reg: {running_reg_unb:.6f} - running_acc: {running_acc_unb:.4f} - lr: {scheduler.get_last_lr()[0]:.8f}', end='')

          scheduler.step()
      
      epoch_loss = np.mean(epoch_loss)
      epoch_acc  = np.mean(epoch_acc)
      epoch_reg  = np.mean(epoch_reg)
      
      print(f' - epoch_loss: {epoch_loss:.4f} - epoch_reg: {epoch_reg:.6f} - epoch_acc: {epoch_acc:.4f}', end='')

      epoch_loss, epoch_acc, epoch_reg = [], [], []

      
      if epoch >= skip_eval:
        model.eval()

        valid_datagen = iter(valid_loader)
        
        with torch.no_grad():
          for i, data in enumerate(valid_datagen):
            inputs, labels, mask = data['input_ids'], data['label'], data['attention_mask']
            inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)

            outputs, aux_losses = model(inputs, mask)
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
        save_model(model, optimizer, str(epoch) + '_' + name)
      
      else:
        epoch_loss, epoch_acc, epoch_reg = 0.0, 0.0, 0.0

      #epoch computing time
      t = time.time() - t

      print(f' - valid_loss: {epoch_loss:.4f} - valid_reg: {epoch_reg:.6f} - valid_acc: {epoch_acc:.4f} - epoch_time: {t:.4f} s')
 
  checkpoint = torch.load(name)
  return checkpoint

def test_ag_model(SETUP, model, criterion, test_dataset):
  epoch_loss, epoch_acc, epoch_reg = [], [], []

  device = SETUP['device']
  BATCH_SIZE = SETUP['batch_size']

  model.eval()
  test_datagen = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
  test_datagen = iter(test_datagen)

  process_inputs = lambda x: torch.Tensor(x.numpy()).to(torch.int64)

  t = time.time()

  with torch.no_grad():
    for i, data in enumerate(test_datagen):
      inputs, labels, mask = data['input_ids'], data['label'], data['attention_mask']
      inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)

      outputs, aux_losses = model(inputs, mask)
      loss = criterion(outputs, labels)
      acc = accuracy(outputs, labels)
      aux_losses = sum(aux_losses) if aux_losses else torch.Tensor([ 0.0 ]).to(device)

      epoch_loss.append(loss.item())
      epoch_acc.append(acc)
      epoch_reg.append(aux_losses.item())

  t = time.time() - t

  epoch_loss, epoch_acc, epoch_reg = np.mean(epoch_loss), np.mean(epoch_acc), np.mean(epoch_reg)

  print(f' - test_loss: {epoch_loss:.4f} - test_reg: {epoch_reg:.6f} - test_acc: {epoch_acc:.4f} - test_time: {t:.4f} s')
  return epoch_loss, epoch_reg, epoch_acc