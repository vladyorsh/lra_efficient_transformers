from lra.layers import *
from lra.models import *
from lra.setups import *
from lra.utils  import *

import sys
import os
import argparse

from lra_benchmarks.matching.input_pipeline import get_matching_datasets

#TODO:
#0.5 reg weight for matching
#redo with lightning cli
#get rid of setups

#Args:
#task
#length
#model

#batch_size
#precision

#lra_lib_path
#lra_data_path

#device
#n devices
#

def get_lra_data(lib_path, data_path, task, batch_size, max_length):
    sys.path.append(os.path.realpath(lib_path))
        
    from lra_benchmarks.matching.input_pipeline import get_matching_datasets
    from lra_benchmarks.listops.input_pipeline import get_datasets as get_listops_datasets
    from lra_benchmarks.text_classification.input_pipeline import get_tc_datasets
    
    MATCHING_DATADIR = os.path.join(os.path.realpath(data_path), 'lra_release/lra_release/tsv_data')
    LISTOPS_DATADIR  = os.path.join(os.path.realpath(data_path), 'lra_release/listops-1000/')
    
    DATASETS_BY_TASK = {
        'classification' : lambda batch_size, max_length: get_tc_datasets(1, 'imdb_reviews', batch_size=batch_size, max_length=max_length),
        'matching'       : lambda batch_size, max_length: get_matching_datasets(1, None, tokenizer='char', data_dir=MATCHING_DATADIR, batch_size=batch_size, max_length=max_length),
        'listops'        : lambda batch_size, max_length: get_listops_datasets(1, 'basic', data_dir=LISTOPS_DATADIR, batch_size=batch_size, max_length=max_length),
    }
    
    return DATASETS_BY_TASK[task.lower()](batch_size, max_length)
        
def get_model(task, length, model):
    BASE_MODELS = { 'classification' : ClassificationTransformer, 'matching' : MatchingTransformer }
    LUNA_MODELS = { 'classification' : LunaClassifier,            'matching' : LunaMatcher }
    
    REGISTERED_MODELS = {
        'base' : BASE_MODELS,
        'luna' : LUNA_MODELS,
    }
    
    REGISTERED_SETUPS = {
        'classification' : CLS_SETUP,
        'matching' : MATCHING_SETUP,
        'listops' : LISTOPS_SETUP,
    }
    
    setup = REGISTERED_SETUPS[task]
    task = 'classification' if task in { 'classification', 'listops' } else 'matching'
    model_class = REGISTERED_MODELS[model][task]
    model = LraLightningWrapper(
        model_class(
            classes=setup['classes'],
            num_embeddings=setup['num_embeddings'],
            seq_len=length,
            hidden_dim=setup['hidden_dim'],
            qkv_dim=setup['qkv_dim'],
            mlp_dim=setup['mlp_dim'],
            num_heads=setup['num_heads'],
            num_blocks=setup['num_blocks'],
            output_mlp_units=setup['output_mlp_units'],
            internal_dropout_rate=setup['internal_dropout_rate'],
            output_dropout_rate=setup['output_dropout_rate'],
            affine=setup['affine'],
        ),
        reg_weight=1.0,
        betas=(0.9, 0.98),
        base_lr=setup['lr'],
        wd=setup['weight_decay'],
        schedule=setup['schedule'](),
    )
    
    return model

def main(args):
    task, max_length, model, batch_size, fp16, devices, lib_path, data_path = args.task, args.max_length, args.model, args.batch_size, args.fp16, args.devices, args.lib_path, args.data_path
    train_dataset, valid_dataset, test_dataset, encoder = get_lra_data(lib_path, data_path, task, batch_size, max_length)
    train_dataset, valid_dataset, test_dataset = torch_generator_wrapper(train_dataset), torch_generator_wrapper(valid_dataset), torch_generator_wrapper(test_dataset)
    
    model = get_model(task, model)
    trainer = Trainer()