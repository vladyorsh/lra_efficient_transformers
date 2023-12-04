from .layers import *
from .utils import LossMetric, MyAcc
from collections.abc import Iterable
import torch
import torchmetrics
import lightning.pytorch as pl
import matplotlib.pyplot as plt

class Encoder(nn.Module):
  def __init__(self, module_type, num_blocks, *args, **kwargs):
    super(Encoder, self).__init__()
    
    self.blocks = nn.ModuleList([ module_type(* args, ** kwargs) for _ in range(num_blocks) ])
    
  def forward(self, x, mask=None, losses=[], artifacts=[]):
    for block in self.blocks:
        if torch.is_tensor(x):
            x = block(x, mask, losses=losses, artifacts=artifacts)
        else:
            x = block(* x, mask, losses=losses, artifacts=artifacts)
    return x

class ClassificationTransformer(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, use_cls=True, logging_frequency=1000, norm_type='layernorm'):
    super(ClassificationTransformer, self).__init__()
    
    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len, use_cls=use_cls)
    self.encoder     = Encoder(TBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine, logging_frequency, norm_type)
    self.classifier  = TClassifier(classes, hidden_dim, mlp_dim, output_dropout_rate, affine, use_cls, norm_type)
    self.logging_frequency = logging_frequency

  def forward(self, inputs, mask=None):
    additional_losses = []
    artifacts = []

    x, mask = self.embed_layer(inputs, mask)
    x = self.encoder(x, mask, losses=additional_losses, artifacts=artifacts)
    x = self.classifier(x, mask)

    return x, additional_losses, artifacts

class LunaClassifier(ClassificationTransformer):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, use_cls=True, logging_frequency=1000, norm_type='layernorm', mem_size=256, shared_att='full'):
    super(LunaClassifier, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, use_cls, logging_frequency, norm_type)

    self.encoder     = Encoder(LunaBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine, logging_frequency, norm_type, shared_att)
    self.mem         = nn.Parameter(torch.empty(1, mem_size, hidden_dim), requires_grad=True)
    nn.init.normal_(self.mem)
    
  def forward(self, inputs, mask):
    mem = self.mem
    losses = []
    artifacts = []
    
    x, mask= self.embed_layer(inputs, mask)
    x, mem = self.encoder((x, mem), mask, losses, artifacts)
    x      = self.classifier(x, mask)

    return x, losses, artifacts
    
class vMFLunaClassifier(LunaClassifier):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, use_cls=True, logging_frequency=1000, norm_type='layernorm', mem_size=256, shared_att='full', vmf_k=10.0):
    super(vMFLunaClassifier, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, use_cls, logging_frequency, norm_type, mem_size, shared_att)

    self.encoder     = Encoder(vMFLunaBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine, logging_frequency, shared_att, vmf_k, mem_size)
    
class ConvLunaClassifier(LunaClassifier):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, use_cls=True, logging_frequency=1000, norm_type='layernorm', mem_size=256, shared_att='full', kernel=(4, 1), stride=(1, 1), pool=False):
    super(ConvLunaClassifier, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, use_cls, logging_frequency, norm_type, mem_size, shared_att)

    self.encoder     = Encoder(ConvLunaBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine, logging_frequency, shared_att, kernel, stride, None, pool)
    
class BLunaClassifier(LunaClassifier):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, use_cls=True, logging_frequency=1000, norm_type='layernorm', mem_size=256, shared_att='full', weibull_k=10.0, gamma_beta=1e-4, prior_hidden_size=32, anneal_k=0.00015, anneal_b=6.25, eps=1e-5):
    super(BLunaClassifier, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, use_cls, logging_frequency, norm_type, mem_size, shared_att)

    self.encoder     = Encoder(BLunaBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine, logging_frequency, shared_att, weibull_k, gamma_beta, prior_hidden_size, anneal_k, anneal_b, eps)
    
class MatchingTransformer(ClassificationTransformer):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, use_cls=True, logging_frequency=1000):
    super(MatchingTransformer, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, use_cls, logging_frequency)
    self.classifier  = DualClassifier(classes, hidden_dim, mlp_dim, affine, use_cls)

  def forward(self, inputs, masks):
    additional_losses = []
    artifacts_1 = []
    artifacts_2 = []
    
    if masks is not None:
        mask_1, mask_2 = masks
    else:
        mask_1, mask_2 = None, None

    emb_1, mask_1 = self.embed_layer(inputs[0], mask_1)
    emb_2, mask_2 = self.embed_layer(inputs[1], mask_2)
    
    emb_1 = self.encoder(emb_1, mask_1, losses=additional_losses, artifacts=artifacts_1)
    emb_2 = self.encoder(emb_2, mask_2, losses=additional_losses, artifacts=artifacts_2)
    
    x = self.classifier((emb_1, emb_2))
    
    artifacts = list(zip(artifacts_1, artifacts_2))

    return x, additional_losses, artifacts

class LunaMatcher(LunaClassifier):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, use_cls=True, logging_frequency=1000, mem_size=256, shared_att='full'):
    super(LunaMatcher, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine, use_cls, logging_frequency, mem_size, shared_att)
    self.classifier  = DualClassifier(classes, hidden_dim, mlp_dim, affine, use_cls)
    
  def forward(self, inputs, masks):
    mem_1, mem_2 = self.mem, self.mem
    additional_losses = []
    artifacts_1 = []
    artifacts_2 = []
    
    if masks is not None:
        mask_1, mask_2 = masks
    else:
        mask_1, mask_2 = None, None

    emb_1, mask_1 = self.embed_layer(inputs[0], mask_1)
    emb_2, mask_2 = self.embed_layer(inputs[1], mask_2)
    
    emb_1, mem_1 = self.encoder((emb_1, mem_1), mask_1, losses=additional_losses, artifacts=artifacts_1)
    emb_2, mem_2 = self.encoder((emb_2, mem_2), mask_2, losses=additional_losses, artifacts=artifacts_2)
    
    x = self.classifier((emb_1, emb_2))
    
    artifacts = list(zip(artifacts_1, artifacts_2))

    return x, additional_losses, artifacts
    
class LraLightningWrapper(pl.LightningModule):
    def __init__(self, model, reg_weight=1.0, betas=(0.9, 0.98), base_lr=0.05, wd=0.1, schedule=lambda x: 1.0, log_non_scalars=True, log_params=True, mask_inputs=False):
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
        self.log_params = log_params
        self.mask_inputs = mask_inputs
        
        #nn.ModuleDict is needed for correct handling of multi-device training
        self.train_metrics = nn.ModuleDict({
            'loss'     : LossMetric(),
            'reg_loss' : LossMetric(),
            'accuracy' : torchmetrics.classification.MulticlassAccuracy(self.model.classifier.classes),
            'my_acc'   : MyAcc(self.model.classifier.classes),
        })
        
        self.test_metrics = nn.ModuleDict({
            'loss'     : LossMetric(),
            'reg_loss' : LossMetric(),
            'accuracy' : torchmetrics.classification.MulticlassAccuracy(self.model.classifier.classes),
            'my_acc'   : MyAcc(self.model.classifier.classes),
        })
        
        self.logged_first_iter = False
    
        
    def on_train_start(self):
        for name, metric in self.train_metrics.items():
            metric.reset()
        for name, metric in self.test_metrics.items():
            metric.reset()
            
    def log_artifact(self, artifact, type, name):
        exp = self.logger.experiment
        name= type + '_' + name
        
        scale = 6
        max_size = 16
        w, h = scale, scale
        
        if type == 'tensor_slice':
            if artifact.shape[0] > artifact.shape[1]:
                w = scale
                h = round(artifact.shape[0] / artifact.shape[1] * scale)
                if h > max_size:
                    h, w = max_size, w * max_size / h
            else:
                h = scale
                w = round(artifact.shape[1] / artifact.shape[0] * scale)
                if w > max_size:
                    h, w = h * max_size / w, max_size
                    
            plt.figure(figsize=(w, h))
            plt.colorbar(plt.imshow(artifact))
            exp.add_figure(name, plt.gcf(), global_step=self.trainer.global_step, close=True)
        elif type == 'tensor_stack':
            batch_size = artifact.shape[0]
            ncols=batch_size
            nrows=1
            while ncols > 4 and ncols > 2 * nrows and not (ncols % 2):
                ncols //= 2
                nrows *= 2
            fig, ax = plt.subplots(figsize=(ncols * scale, nrows * scale), ncols=ncols, nrows=nrows)
            for axis, image in zip(ax.flat, artifact):
                plt.colorbar(axis.imshow(image))
            exp.add_figure(name, fig, global_step=self.trainer.global_step, close=True)
            
        elif type == 'hist':
            plt.hist(artifact.flatten(), bins=50, edgecolor='black')
            plt.grid()
            exp.add_figure(name, plt.gcf(), global_step=self.trainer.global_step, close=True)
        else:
            raise NotImplementedError(f'Other artifact types such as {type} are not yet supported!')
            
    def log_artifacts(self, artifacts, prefix=''):
        for i, item in enumerate(artifacts):
            if isinstance(item, Iterable):
                self.log_artifacts(item, prefix=f'{i}_')
            else:
                artifact, name, type, log_every = item.artifact, item.name, item.type, item.log_every
                if not (self.trainer.global_step % log_every):
                    name = prefix + name
                    if not isinstance(type, str):
                        for t in type: self.log_artifact(artifact, t, name)
                    else:
                        self.log_artifact(artifact, type, name)
    
    def prepare_tensor_for_viz(self, artifact):
        artifact = artifact.detach()
        if len(artifact.shape) > 2:
            artifact = artifact[0]
        if len(artifact.shape) < 2:
            artifact = artifact.unsqueeze(0)
        if artifact.shape[0] < 10 * artifact.shape[1]:
            aspect_ratio = max(1, round(artifact.shape[1] / artifact.shape[0] / 2))
            artifact = artifact.repeat_interleave(aspect_ratio, dim=0)
        return artifact
    
    def log_self_params(self, artifacts, types=('tensor_slice', 'hist')):
        for name, param in self.model.named_parameters():
            p = param.data.squeeze()
            while len(p.shape) < 2:
                p = p.unsqueeze(0)
            artifact = self.prepare_tensor_for_viz(p)
            artifacts.append(
                Artifact(artifact, name, types, self.model.logging_frequency)
            )
                
    def log_self_grads(self, artifacts, types=('tensor_slice', 'hist')):
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                artifact = self.prepare_tensor_for_viz(param.grad)                        
                artifacts.append(
                    Artifact(artifact, name + '_grad', types, self.model.logging_frequency)
                )
    
    def unpack_batch(self, batch):
        batch_size=None
        if 'inputs' in batch.keys():
            inp = torch.from_numpy(batch['inputs']).to(self.device)
            batch_size = inp.shape[0]
            mask = (inp != 0).float()
        else:
            inp = (torch.from_numpy(batch['inputs1']).to(self.device), torch.from_numpy(batch['inputs2']).to(self.device))
            batch_size = inp[0].shape[0]
            mask = ((inp[0] != 0).float(), (inp[1] != 0).float())
        target = torch.from_numpy(batch['targets']).long().to(self.device)
        if not self.mask_inputs:
            mask = None
        return inp, target, mask, batch_size
        
    def on_after_backward(self):
        artifacts = []
        if self.log_non_scalars:
            if self.log_params:
                self.log_self_grads(artifacts, types=('tensor_slice', 'hist'))
        self.log_artifacts(artifacts)
        
    def training_step(self, batch, batch_idx):
        inp, target, mask, batch_size = self.unpack_batch(batch)
        preds, auxiliary_losses, artifacts = self.model(inp, mask)
        
        auxiliary_losses = torch.mean(torch.as_tensor(auxiliary_losses)) if auxiliary_losses else torch.tensor(0.0)
        loss = self.loss(preds, target)
        
        #Logging
        
        #Non-scalar
        if self.log_non_scalars or not self.logged_first_iter:
            if self.log_params or not self.logged_first_iter:
                self.log_self_params(artifacts, types=('tensor_slice', 'hist'))
            self.log_artifacts(artifacts)
            self.logged_first_iter = True
                    
        
        #Metrics
        self.train_metrics['loss'](loss)
        self.train_metrics['reg_loss'](auxiliary_losses)
        
        self.log("loss_step",     self.train_metrics['loss'], prog_bar=True)
        self.log("reg_loss_step", self.train_metrics['reg_loss'], prog_bar=False)
        
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
        inp, target, mask, batch_size = self.unpack_batch(batch)
        preds, auxiliary_losses, _ = self.model(inp, mask)
        
        auxiliary_losses = torch.mean(torch.as_tensor(auxiliary_losses)) if auxiliary_losses else torch.tensor(0.0)
        loss = self.loss(preds, target)
        
        #Logging
        self.test_metrics['loss'](loss)
        self.test_metrics['reg_loss'](auxiliary_losses)
            
        self.log('val_loss', self.test_metrics['loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, prog_bar=True)
        self.log('val_reg_loss', self.test_metrics['reg_loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, prog_bar=False)
        
        for name, metric in self.test_metrics.items():
            if name in { 'loss', 'reg_loss' }: continue
            metric(preds, target)
            self.log('val_' + name, metric, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, prog_bar=True)
        
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
        inp, target, mask, batch_size = self.unpack_batch(batch)
        preds, auxiliary_losses, _ = self.model(inp, mask)
        
        auxiliary_losses = torch.mean(torch.as_tensor(auxiliary_losses)) if auxiliary_losses else torch.tensor(0.0)
        loss = self.loss(preds, target)
        
        #Logging
        self.test_metrics['loss'](loss)
        self.test_metrics['reg_loss'](auxiliary_losses)
            
        self.log('test_loss', self.test_metrics['loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, prog_bar=True)
        self.log('test_reg_loss', self.test_metrics['reg_loss'], on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, prog_bar=False)
        
        for name, metric in self.test_metrics.items():
            if name in { 'loss', 'reg_loss' }: continue
            metric(preds, target)
            self.log('test_' + name, metric, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size, prog_bar=True)
        
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