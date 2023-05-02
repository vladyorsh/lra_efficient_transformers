from .layers import *

class ClassificationTransformer(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0):
    super(ClassificationTransformer, self).__init__()
    
    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len)
    self.blocks      = nn.ModuleList([ TBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate) for _ in range(num_blocks) ])
    self.classifier  = TClassifier(classes, hidden_dim, output_mlp_units, output_dropout_rate)

  def forward(self, pixel_values):
    additional_losses = []

    x = self.embed_layer(pixel_values)
    
    for block in self.blocks:
      x = block(x, losses=additional_losses)
    
    x = self.classifier(x)

    return x, additional_losses

class LunaClassifier(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0, mem_size=256):
    super(LunaClassifier, self).__init__()

    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len)
    self.blocks      = nn.ModuleList([ LunaBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate) for _ in range(num_blocks) ])
    self.classifier  = TClassifier(classes, hidden_dim, output_mlp_units, output_dropout_rate)

    self.mem = nn.Parameter(torch.empty(1, mem_size, hidden_dim), requires_grad=True)
    nn.init.xavier_uniform_(self.mem)
    #self.mem.data = self.mem.data.unsqueeze(0)
    
  def forward(self, input):
    x = self.embed_layer(input)
    mem = self.mem
    losses = []

    for block in self.blocks:
      x, mem = block(x, mem, losses)

    x = self.classifier(x)

    return x, losses

class MatchingTransformer(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0, mem_size=256):
    super(MatchingTransformer, self).__init__()
    
    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len)
    self.blocks      = nn.ModuleList([ TBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate) for _ in range(num_blocks) ])
    self.classifier  = DualClassifier(classes, hidden_dim, output_mlp_units)

  def forward(self, inputs):
    additional_losses = []

    emb_1 = self.embed_layer(inputs[0])
    emb_2 = self.embed_layer(inputs[1])

    for block in self.blocks:
      emb_1 = block(emb_1, losses=additional_losses)
      emb_2 = block(emb_2, losses=additional_losses)
    
    x = self.classifier((emb_1, emb_2))

    return x, additional_losses

class LunaMatcher(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0, mem_size=256):
    super(LunaMatcher, self).__init__()

    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len)
    self.blocks      = nn.ModuleList([ LunaBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate) for _ in range(num_blocks) ])
    self.classifier  = DualClassifier(classes, hidden_dim, output_mlp_units)

    self.mem = nn.Parameter(torch.empty(1, mem_size, hidden_dim), requires_grad=True)
    nn.init.xavier_uniform_(self.mem)
    #self.mem.data = self.mem.data.unsqueeze(0)
    
  def forward(self, inputs):
    mem = self.mem
    additional_losses = []

    emb_1 = self.embed_layer(inputs[0])
    emb_2 = self.embed_layer(inputs[1])

    for block in self.blocks:
      emb_1 = block(emb_1, mem, losses=additional_losses)
      emb_2 = block(emb_2, mem, losses=additional_losses)
    
    x = self.classifier((emb_1, emb_2))

    return x, additional_losses

class ClassificationTransformerSkip(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0):
    super(ClassificationTransformerSkip, self).__init__()
    
    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len)
    self.blocks      = nn.ModuleList([ SBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate) for _ in range(num_blocks) ])
    self.classifier  = TClassifier(classes, hidden_dim, output_mlp_units, output_dropout_rate)

  def forward(self, pixel_values):
    additional_losses = []

    x = self.embed_layer(pixel_values)
    
    for block in self.blocks:
      x = block(x, additional_losses)
    
    x = self.classifier(x)

    return x, additional_losses

class MatchingTransformerSkip(nn.Module):
  def __init__(self, classes, num_embeddings, seq_len, hidden_dim, qkv_dim, mlp_dim, num_heads, num_blocks, output_mlp_units, internal_dropout_rate=0.1, output_dropout_rate=0.0):
    super(MatchingTransformerSkip, self).__init__()
    
    self.embed_layer = TEmbedding(num_embeddings, hidden_dim, seq_len)
    self.blocks      = nn.ModuleList([ SBlock(hidden_dim, qkv_dim, mlp_dim, num_heads, internal_dropout_rate) for _ in range(num_blocks) ])
    self.classifier  = DualClassifier(classes, hidden_dim, output_mlp_units)

  def forward(self, inputs):
    additional_losses = []

    emb_1 = self.embed_layer(inputs[0])
    emb_2 = self.embed_layer(inputs[1])

    for block in self.blocks:
      emb_1 = block(emb_1, additional_losses)
      emb_2 = block(emb_2, additional_losses)
    
    x = self.classifier((emb_1, emb_2))

    return x, additional_losses