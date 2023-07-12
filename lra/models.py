from .layers import *
import torch
import lightning.pytorch as pl
import torchmetrics

class Encoder(nn.Module):
  def __init__(self, module_type, num_blocks, *args, **kwargs):
    super(Encoder, self).__init__()
    
    self.blocks = nn.ModuleList([ module_type(* args, ** kwargs) for _ in range(num_blocks) ])
    
  def forward(self, x, losses=[]):
    for block in self.blocks:
        if torch.is_tensor(x):
            x = block(x, losses=losses)
        else:
            x = block(* x, losses=losses)
    return x

class ClassificationTransformer(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True):
    super(ClassificationTransformer, self).__init__()
    
    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len)
    self.encoder     = Encoder(TBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine)
    self.classifier  = TClassifier(classes, hidden_dim, mlp_dim, output_dropout_rate, affine)

  def forward(self, pixel_values):
    additional_losses = []

    x = self.embed_layer(pixel_values)
    x = self.encoder(x, additional_losses)
    x = self.classifier(x)

    return x, additional_losses

class LunaClassifier(ClassificationTransformer):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, mem_size=256):
    super(LunaClassifier, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine)

    self.encoder     = Encoder(LunaBlock, num_blocks, hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, affine)    
    self.mem         = nn.Parameter(torch.empty(1, mem_size, hidden_dim), requires_grad=True)
    nn.init.normal_(self.mem)
    
  def forward(self, input):
    mem = self.mem
    losses = []
    x      = self.embed_layer(input)
    x, mem = self.encoder((x, mem), losses)
    x      = self.classifier(x)

    return x, losses

class MatchingTransformer(ClassificationTransformer):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True):
    super(MatchingTransformer, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine)
    self.classifier  = DualClassifier(classes, hidden_dim, mlp_dim, affine)

  def forward(self, inputs):
    additional_losses = []

    emb_1 = self.embed_layer(inputs[0])
    emb_2 = self.embed_layer(inputs[1])

    emb_1 = self.encoder(emb_1, losses=additional_losses)
    emb_2 = self.encoder(emb_2, losses=additional_losses)
    
    x = self.classifier((emb_1, emb_2))

    return x, additional_losses

class LunaMatcher(MatchingTransformer):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate=0.1, output_dropout_rate=0.0, affine=True, mem_size=256):
    super(LunaMatcher, self).__init__(classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, internal_dropout_rate, output_dropout_rate, affine)
    self.classifier  = DualClassifier(classes, hidden_dim, affine)
    
  def forward(self, inputs):
    mem_1, mem_2 = self.mem, self.mem
    additional_losses = []

    emb_1 = self.embed_layer(inputs[0])
    emb_2 = self.embed_layer(inputs[1])

    emb_1, mem_1 = self.encoder((emb_1, mem_1), losses=additional_losses)
    emb_2, mem_2 = self.encoder((emb_2, mem_2), losses=additional_losses)
    
    x = self.classifier((emb_1, emb_2))

    return x, additional_losses
    
class LraLightningWrapper(pl.LightningModule):
    def __init__(self, model, reg_weight=1.0, betas=(0.9, 0.98), base_lr=0.05, wd=0.1, schedule=lambda x: 1.0):
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.loss  = nn.CrossEntropyLoss()
        
        self.reg_weight = reg_weight
        self.betas = betas
        self.base_lr = base_lr
        self.wd = wd
        self.schedule = schedule
        
        #nn.ModuleDict is needed for correct handling of multi-device training
        self.train_metrics = nn.ModuleDict({
            'accuracy' : torchmetrics.classification.Accuracy(),
        })
        
        self.test_metrics = nn.ModuleDict({
            'accuracy' : torchmetrics.classification.Accuracy(),
        })
            
    def training_step(self, batch, batch_idx):
        inp, target = batch
        preds, auxiliary_losses = self.model(inp)
        
        auxiliary_losses = torch.mean(auxiliary_losses) if auxiliary_losses else 0.0
        loss = self.loss(preds, target)
        
        #Logging
        #TODO: Check for the correct reset at the eval period end
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("reg_loss", auxiliary_losses, on_step=True, on_epoch=True)
        
        for name, metric in self.train_metrics.items():
            metric(outputs['preds'], outputs['target'])
            self.log(name, metric, on_step=True, on_epoch=True, prog_bar=True)
        
        loss = loss + auxiliary_losses * self.reg_weight    
        
        return {'loss' : loss, 'preds' : preds, 'target' : target}
            
    def validation_step(self, batch, batch_idx):
        inp, target = batch
        preds, auxiliary_losses = self.model(inp)
        
        auxiliary_losses = torch.mean(auxiliary_losses) if auxiliary_losses else 0.0
        loss = self.loss(preds, target)
        
        #Logging
        #TODO: Check for the correct reset at the eval period end
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_reg_loss", auxiliary_losses, on_step=False, on_epoch=True)
        
        for name, metric in self.test_metrics.items():
            name = 'valid_' + name
            metric(outputs['preds'], outputs['target'])
            self.log(name, metric, on_step=False, on_epoch=True, prog_bar=True)
        
        loss = loss + auxiliary_losses * self.reg_weight    
        
        return {'loss' : loss, 'preds' : preds, 'target' : target}
        

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