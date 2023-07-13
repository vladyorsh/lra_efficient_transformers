from .layers import *
import torch
import lightning.pytorch as pl
import torchmetrics
from collections.abc import Iterable

class Encoder(nn.Module):
  def __init__(self, module_type, num_blocks, *args, **kwargs):
    super(Encoder, self).__init__()
    
    self.blocks = nn.ModuleList([ module_type(* args, ** kwargs) for _ in range(num_blocks) ])
    
  def forward(self, x, losses=[], artifacts=[]):
    for block in self.blocks:
        if torch.is_tensor(x):
            x = block(x, losses=losses, artifacts=artifacts)
        else:
            x = block(* x, losses=losses, artifacts=artifacts)
    return x

class ClassificationTransformer(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, logging_frequency=1000):
    super(ClassificationTransformer, self).__init__()
    
    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len)
    self.encoder     = Encoder(TBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine, logging_frequency)
    self.classifier  = TClassifier(classes, hidden_dim, mlp_dim, output_dropout_rate, affine)

  def forward(self, pixel_values):
    additional_losses = []
    artifacts = []

    x = self.embed_layer(pixel_values)
    x = self.encoder(x, additional_losses, artifacts)
    x = self.classifier(x)

    return x, additional_losses, artifacts

class LunaClassifier(ClassificationTransformer):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, logging_frequency=1000, mem_size=256, shared_att='full'):
    super(LunaClassifier, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, logging_frequency)

    self.encoder     = Encoder(LunaBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine, logging_frequency, shared_att)
    self.mem         = nn.Parameter(torch.empty(1, mem_size, hidden_dim), requires_grad=True)
    nn.init.normal_(self.mem)
    
  def forward(self, input):
    mem = self.mem
    losses = []
    artifacts = []
    
    x      = self.embed_layer(input)
    x, mem = self.encoder((x, mem), losses, artifacts)
    x      = self.classifier(x)

    return x, losses, artifacts

class MatchingTransformer(ClassificationTransformer):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, logging_frequency=1000):
    super(MatchingTransformer, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, logging_frequency)
    self.classifier  = DualClassifier(classes, hidden_dim, mlp_dim, affine)

  def forward(self, inputs):
    additional_losses = []
    artifacts_1 = []
    artifacts_2 = []

    emb_1 = self.embed_layer(inputs[0])
    emb_2 = self.embed_layer(inputs[1])

    emb_1 = self.encoder(emb_1, losses=additional_losses, artifacts=artifacts_1)
    emb_2 = self.encoder(emb_2, losses=additional_losses, artifacts=artifacts_2)
    
    x = self.classifier((emb_1, emb_2))
    
    artifacts = list(zip(artifacts_1, artifacts_2))

    return x, additional_losses, artifacts

class LunaMatcher(LunaClassifier):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, logging_frequency=1000, mem_size=256):
    super(LunaMatcher, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, logging_frequency, mem_size)
    self.classifier  = DualClassifier(classes, hidden_dim, affine)
    
  def forward(self, inputs):
    mem_1, mem_2 = self.mem, self.mem
    additional_losses = []
    artifacts_1 = []
    artifacts_2 = []


    emb_1 = self.embed_layer(inputs[0])
    emb_2 = self.embed_layer(inputs[1])

    emb_1, mem_1 = self.encoder((emb_1, mem_1), losses=additional_losses, artifacts=artifacts_1)
    emb_2, mem_2 = self.encoder((emb_2, mem_2), losses=additional_losses, artifacts=artifacts_2)
    
    x = self.classifier((emb_1, emb_2))
    
    artifacts = list(zip(artifacts_1, artifacts_2))

    return x, additional_losses, artifacts
    
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
    
class LraLightningWrapper(pl.LightningModule):
    def __init__(self, model, reg_weight=1.0, betas=(0.9, 0.98), base_lr=0.05, wd=0.1, schedule=lambda x: 1.0, log_non_scalars=True):
        super().__init__()
        #self.automatic_optimization = False
        self.model = model
        self.loss  = nn.CrossEntropyLoss()
        
        self.reg_weight = reg_weight
        self.betas = betas
        self.base_lr = base_lr
        self.wd = wd
        self.schedule = schedule
        self.log_non_scalars = log_non_scalars
        
        #nn.ModuleDict is needed for correct handling of multi-device training
        self.train_metrics = nn.ModuleDict({
            'loss'     : LossMetric(),
            'reg_loss' : LossMetric(),
            'accuracy' : torchmetrics.classification.MulticlassAccuracy(self.model.classifier.classes),
        })
        
        self.test_metrics = nn.ModuleDict({
            'loss'     : LossMetric(),
            'reg_loss' : LossMetric(),
            'accuracy' : torchmetrics.classification.MulticlassAccuracy(self.model.classifier.classes),
        })
    
        
    def on_train_start(self):
        for name, metric in self.train_metrics.items():
            metric.reset()
        for name, metric in self.test_metrics.items():
            metric.reset()
            
    def log_artifact(self, artifact, type, name):
        exp = self.logger.experiment
        name= type + '_' + name
        if type == 'tensor_slice':
            plt.imshow(artifact)
            exp.add_figure(name, plt.gcf(), global_step=self.trainer.global_step, close=True)
        elif type == 'hist':
            plt.hist(artifact.flatten(), bins=50, edgecolor='black')
            plt.grid()
            exp.add_figure(name, plt.gcf(), global_step=self.trainer.global_step, close=True)
        else:
            raise NotImplementedError('Other artifact types are not yet supported!')
            
    def log_artifacts(self, artifacts, prefix=''):
        for i, item in enumerate(artifacts):
            if isinstance(item, Iterable):
                self.log_artifacts(item, prefix=f'{i}_')
            else:
                artifact, name, type, log_every = item.artifact, item.name, item.type, item.log_every
                if not (self.trainer.global_step % log_every):
                    name = prefix + name
                    if isinstance(type, Iterable):
                        for t in type: self.log_artifact(artifact, t, name)
                    else:
                        self.log_artifact(artifact, type, name)
        
    def training_step(self, batch, batch_idx):
        inp, target = torch.from_numpy(batch['inputs']).to(self.device), torch.from_numpy(batch['targets']).to(self.device)
        preds, auxiliary_losses, artifacts = self.model(inp)
        
        auxiliary_losses = torch.mean(auxiliary_losses) if auxiliary_losses else torch.tensor(0.0)
        loss = self.loss(preds, target)
        
        #Logging
        if self.log_non_scalars:
            self.log_artifacts(artifacts)
        
        self.train_metrics['loss'](loss)
        self.train_metrics['reg_loss'](auxiliary_losses)
        
        self.log("loss_step",     self.train_metrics['loss'], prog_bar=True)
        self.log("reg_loss_step", self.train_metrics['reg_loss'], prog_bar=True)
        
        for name, metric in self.train_metrics.items():
            if name in { 'loss', 'reg_loss' }: continue
            metric(preds, target)
            self.log(name + '_step', metric, prog_bar=True)
        
        loss = loss + auxiliary_losses * self.reg_weight    
        
        return loss
        
    #Validation logging logic is meant to be used with completing only a set number of train steps per "epoch" but with the complete validation over the whole dataset
    def on_validation_start(self):
        #Logging the epoch training metrics since the "epoch" ends there
        for name, metric in self.train_metrics.items():
            name = name + '_epoch'
            self.log(name, metric.compute(), sync_dist=True, prog_bar=True)
        #Preparing fresh metrics for validation
        for name, metric in self.test_metrics.items():
            metric.reset()
            
    def validation_step(self, batch, batch_idx):
        inp, target = torch.from_numpy(batch['inputs']).to(self.device), torch.from_numpy(batch['targets']).to(self.device)
        preds, auxiliary_losses, _ = self.model(inp)
        
        auxiliary_losses = torch.mean(auxiliary_losses) if auxiliary_losses else torch.tensor(0.0)
        loss = self.loss(preds, target)
        
        #Logging
        self.test_metrics['loss'](loss)
        self.test_metrics['reg_loss'](auxiliary_losses)
            
        self.log('val_loss', self.test_metrics['loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=inp.shape[0], prog_bar=True)
        self.log('val_reg_loss', self.test_metrics['reg_loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=inp.shape[0], prog_bar=True)
        
        for name, metric in self.test_metrics.items():
            if name in { 'loss', 'reg_loss' }: continue
            metric(preds, target)
            self.log('val_' + name, metric, on_step=False, on_epoch=True, sync_dist=True, batch_size=inp.shape[0], prog_bar=True)
        
        loss = loss + auxiliary_losses * self.reg_weight    
        
        return loss
        
    def on_validation_end(self):
        #Clear training metrics since the "epoch" ends there
        for name, metric in self.train_metrics.items():
            metric.reset()
            
    def on_test_start(self):
        for name, metric in self.test_metrics.items():
            metric.reset()
            
    def test_step(self, batch, batch_idx):
        inp, target = torch.from_numpy(batch['inputs']).to(self.device), torch.from_numpy(batch['targets']).to(self.device)
        preds, auxiliary_losses, _ = self.model(inp)
        
        auxiliary_losses = torch.mean(auxiliary_losses) if auxiliary_losses else torch.tensor(0.0)
        loss = self.loss(preds, target)
        
        #Logging
        self.test_metrics['loss'](loss)
        self.test_metrics['reg_loss'](auxiliary_losses)
            
        self.log('test_loss', self.test_metrics['loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=inp.shape[0], prog_bar=True)
        self.log('test_reg_loss', self.test_metrics['reg_loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=inp.shape[0], prog_bar=True)
        
        for name, metric in self.test_metrics.items():
            if name in { 'loss', 'reg_loss' }: continue
            metric(preds, target)
            self.log('test_' + name, metric, on_step=False, on_epoch=True, sync_dist=True, batch_size=inp.shape[0], prog_bar=True)
        
        loss = loss + auxiliary_losses * self.reg_weight    
        
        return loss
        
    #def on_test_end(self):
    #    ...
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.base_lr, weight_decay=self.wd, betas=self.betas)
        
        if self.schedule is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, self.schedule),
                    "monitor": None,
                    "interval" : "step",
                    "frequency": 1,
                    "Name" : "lr",
                    },
                }
        else:
            return optimizer