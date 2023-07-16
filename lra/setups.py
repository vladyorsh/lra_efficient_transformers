from .utils import *

CLS_SETUP = {
    'full_batch_size' : 32,
    
    'lr' : 0.05,
    'weight_decay' : 0.1,
    'steps' : 20000,
    'schedule' : lambda: get_sqrt_schedule(warmup_steps=8000),
    'eval_period' : 200,
    
    'classes' : 2,
    'hidden_dim' : 256,
    'qkv_dim' : 256,
    'mlp_dim': 1024,
    'num_heads' : 4,
    'num_blocks' : 4,
    'output_units' : 1024,
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'patience' : 15,
}

LISTOPS_SETUP = {
    'full_batch_size' : 32,
    
    'lr' : 0.005,
    'weight_decay' : 0.1,
    'schedule' : lambda: get_sqrt_schedule(warmup_steps=1000),
    'steps' : 15000,
    'eval_period' : 50,
    
    'classes' : 10,
    'hidden_dim' : 512,
    'qkv_dim' : 512,
    'num_heads' : 8,
    'num_blocks' : 6,
    'mlp_dim' : 2048,
    'output_units' : 2048,
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'patience' : 150,
}

MATCHING_SETUP = {
    'full_batch_size' : 32,
    
    'lr' : 0.5,
    'weight_decay' : 0.1,
    'schedule' : lambda: get_sqrt_schedule(warmup_steps=8000),
    'steps' : 20000,
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
    
    'patience' : 15,
}