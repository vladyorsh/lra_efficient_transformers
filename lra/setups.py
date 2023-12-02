from .utils import *

CLS_SETUP = {
    'full_batch_size' : 32,
    'max_length' : 4000,
    
    'lr' : 0.005, #Original LRA value 0.05
    'weight_decay' : 0.01, #Original LRA value 0.1
    'steps' : 25000, #Original LRA value 20k
    'schedule' : lambda: get_sqrt_schedule(warmup_steps=8000),
    'eval_period' : 400,
    
    'classes' : 2,
    'hidden_dim' : 256,
    'qkv_dim' : 256,
    'mlp_dim': 1024,
    'num_heads' : 4,
    'num_blocks' : 4,
    'output_units' : 1024,
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'early_stop_patience' : 25,
    'fail_stop_warmup' : 10,
}

LISTOPS_SETUP = {
    'full_batch_size' : 32,
    'max_length' : 2000,
    
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
    
    'early_stop_patience' : 300,
    'fail_stop_warmup' : 50,
}

MATCHING_SETUP = {
    'full_batch_size' : 32,
    'max_length' : 4000,
    
    'lr' : 0.05,
    'weight_decay' : 0.1,
    'schedule' : lambda: get_sqrt_schedule(warmup_steps=8000),
    'steps' : 30000,
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
    
    'early_stop_patience' : 50,
    'fail_stop_warmup' : 10,
}