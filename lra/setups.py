from .utils import *

CLS_SETUP = {
    'full_batch_size' : 32,
    'max_length' : 4000,
    
    'lr' : 0.005, #Original LRA value 0.05
    'weight_decay' : 0.01, #Original LRA value 0.1
    'steps' : 25000, #Original LRA value 20k
    'schedule' : lambda: get_sqrt_schedule(warmup_steps=8000), #Note that this schedule doesn't provide the 1.0 multiplier, rather typically ~0.01
    'eval_period' : 500, #Original LRA vaue 200
    
    'classes' : 2,
    'hidden_dim' : 128,
    'qkv_dim' : 128,
    'mlp_dim': 512,
    'num_heads' : 4,
    'num_blocks' : 4,
    'output_units' : 512,
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'early_stop_patience' : 25,
    'fail_stop_warmup' : 10,
}

LISTOPS_SETUP = {
    'full_batch_size' : 32, #Original 32
    'max_length' : 2000,
    
    'lr' : 0.005, #Original 0.005
    'weight_decay' : 0.01, #Original 0.1
    'schedule' : lambda: get_sqrt_schedule(warmup_steps=1000),
    'steps' : 30000,
    'eval_period' : 200, #Original 50
    
    'classes' : 10,
    'hidden_dim' : 128, #Original 512, try 256
    'qkv_dim' : 128, #Original 512, try 256
    'num_heads' : 4, #Original 8
    'num_blocks' : 6,
    'mlp_dim' : 512, #Original 2048
    'output_units' : 512, #Original 2048
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'early_stop_patience' : 300,
    'fail_stop_warmup' : 50,
}

MATCHING_SETUP = {
    'full_batch_size' : 32, #Original 32, try 64?
    'max_length' : 4000,
    
    'lr' : 0.05, #Original 0.05
    'weight_decay' : 0.04, #Original 0.01
    'schedule' : lambda: get_sqrt_schedule(warmup_steps=8000),
    'steps' : 30000,
    'eval_period' : 500, #Original 200
    'skip_eval' : 0,
    
    'classes' : 2,
    'hidden_dim' : 128,
    'qkv_dim' : 128,
    'mlp_dim' : 512,
    'num_heads' : 4,
    'num_blocks' : 3, #Original 4, try 6?
    'output_units' : 512,
    'internal_dropout_rate' : 0.1,
    'output_dropout_rate' : 0.0,
    
    'early_stop_patience' : 50,
    'fail_stop_warmup' : 10,
}