from .layers import *

class AGTransformer(nn.Module):
  def __init__(self, classes, tokenizer, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0, normalize_len=False):
    super(AGTransformer, self).__init__()
    
    norm_dim = (hidden_dim,) if not normalize_len else (seq_len, hidden_dim)

    self.embed_layer = TEmbedding(tokenizer, hidden_dim, seq_len, internal_dropout_rate, norm_dim)
    self.blocks      = nn.ModuleList([ TBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, norm_dim) for _ in range(num_blocks) ])
    self.pooler      = TPooler(hidden_dim)
    self.classifier  = TClassifier(classes, hidden_dim, output_dropout_rate)

  def forward(self, pixel_values, mask):
    additional_losses = []
    mask = mask.unsqueeze(-1)

    x = self.embed_layer(pixel_values, mask)
    
    for block in self.blocks:
      x = block(x, mask, additional_losses)
    
    #print('-----------------------------')
    x = self.pooler(x)
    x = self.classifier(x)

    return x, additional_losses

class AGTransformerSkip(nn.Module):
  def __init__(self, classes, tokenizer, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0, normalize_len=False):
    super(AGTransformerSkip, self).__init__()
    
    norm_dim = (hidden_dim,) if not normalize_len else (seq_len, hidden_dim)

    self.embed_layer = TEmbedding(tokenizer, hidden_dim, seq_len, internal_dropout_rate, norm_dim)
    self.blocks      = nn.ModuleList([ SBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate, norm_dim) for _ in range(num_blocks) ])
    self.pooler      = TPooler(hidden_dim)
    self.classifier  = TClassifier(classes, hidden_dim, output_dropout_rate)

  def forward(self, pixel_values, mask):
    additional_losses = []
    mask = mask.unsqueeze(-1)

    x = self.embed_layer(pixel_values, mask)
    
    for block in self.blocks:
      x = block(x, mask, additional_losses)
    
    #print('-----------------------------')
    x = self.pooler(x)
    x = self.classifier(x)

    return x, additional_losses