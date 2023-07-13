from lra.models import *
from lra.setups import *
from lra.utils  import *

import sys
import os
import argparse

#TODO:
#0.5 reg weight for matching
#redo with lightning cli
#get rid of setups

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
       
def get_setup(task):
    REGISTERED_SETUPS = {
        'classification' : CLS_SETUP,
        'matching' : MATCHING_SETUP,
        'listops' : LISTOPS_SETUP,
    }
    
    setup = REGISTERED_SETUPS[task]
    return setup
        
def get_model(task, length, setup, model, encoder):
    BASE_MODELS = { 'classification' : ClassificationTransformer, 'matching' : MatchingTransformer }
    LUNA_MODELS = { 'classification' : LunaClassifier,            'matching' : LunaMatcher }
    
    REGISTERED_MODELS = {
        'base' : BASE_MODELS,
        'luna' : LUNA_MODELS,
    }
    
    task = 'classification' if task in { 'classification', 'listops' } else 'matching'
    model_class = REGISTERED_MODELS[model][task]
    model = LraLightningWrapper(
        model_class(
            classes=setup['classes'],
            num_embeddings=encoder.vocab_size,
            seq_len=length,
            hidden_dim=setup['hidden_dim'],
            qkv_dim=setup['qkv_dim'],
            mlp_dim=setup['mlp_dim'],
            num_heads=setup['num_heads'],
            num_blocks=setup['num_blocks'],
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
    
def get_batch_size_and_acc_steps(effective_batch_size, per_device_batch_size, devices, strategy):
    ALLOWED_STRATEGIES = { 'ddp', 'single_tpu', 'single' }
    if strategy not in ALLOWED_STRATEGIES:
        raise ValueError(f'{strategy} strategy is disabled for safety reasons, use strategy from {ALLOWED_STRATEGIES} instead!')
    
    if effective_batch_size % devices:
        raise ValueError('The SETUP BATCH SIZE is not divisible by the DEVICE COUNT for the chosen strategy!')
    if strategy != 'ddp': #Other strategies may split the batch automatically between devices (be careful!)
        sampled_batch_size = per_device_batch_size * devices
    else: #For ddp we sample a full batch
        sampled_batch_size = per_device_batch_size
    if effective_batch_size % sampled_batch_size:
        raise ValueError('The SETUP BATCH SIZE is not divisible by the EFFECTIVE ONE-PASS BATCH SIZE, try to select another per-device batch size!')
    if strategy == 'single':
        strategy = pl.strategies.SingleDeviceStrategy(device=0)
    accumulation_steps = max(1, effective_batch_size // sampled_batch_size)
    return sampled_batch_size, accumulation_steps, strategy

def main(args):
    setup = get_setup(args.task)
    
    #Parse the training strategy and determine the sizes of sampled batches
    sampled_batch_size, accumulation_steps, strategy = get_batch_size_and_acc_steps(setup['full_batch_size'], args.batch_size, args.devices, args.strategy)
    train_dataset, valid_dataset, test_dataset, encoder = get_lra_data(args.lib_path, args.data_path, args.task, sampled_batch_size, args.max_length)
    train_dataset, valid_dataset, test_dataset = wrap_lra_tf_dataset(train_dataset), wrap_lra_tf_dataset(valid_dataset), wrap_lra_tf_dataset(test_dataset)
    
    model = get_model(args.task, args.max_length, setup, args.model, encoder)
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=strategy,
        devices=args.devices,
        num_nodes=1,
        precision=args.precision,
        logger=pl.loggers.CSVLogger("logs", name="my_exp_name"),
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            #pl.callbacks.DeviceStatsMonitor(),
            #pl.callbacks.EarlyStopping(...),
            
            #Progress bars
            #pl.callbacks.RichProgressBar(refresh_rate=1, leave=True),
            #PBar(refresh_rate=50),
            pl.callbacks.TQDMProgressBar(refresh_rate=100),
        ],
        max_steps=setup['steps'],
        val_check_interval=setup['eval_period'],
        accumulate_grad_batches=accumulation_steps,
        #!!!!!!!!
        fast_dev_run=False,
    )
    trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=valid_dataset)


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run LRA tasks with chosen models.')
    parser.add_argument('--task', help='LRA task to be run on')
    parser.add_argument('--max_length', type=int, help='max input length')
    parser.add_argument('--model', help='model architecture')
    parser.add_argument('--batch_size', type=int, help='per-device batch size')
    parser.add_argument('--precision', help='PytorchLightning precision settings for faster computation', default='32-true')
    parser.add_argument('--lib_path', help='relative path to the LRA cloned repo', default='long-range-arena')
    parser.add_argument('--data_path', help='relative path to the LRA unpacked data', default='lra_release')
    parser.add_argument('--accelerator', help='device type to be used', default='gpu')
    parser.add_argument('--devices', help='device count', type=int, default=1)
    parser.add_argument('--strategy', help='distribution strategy', default='ddp')
    parser.add_argument('--data_workers', help='number of DataLoader workers', type=int, default=0)
    
    args = parser.parse_args()
    main(args)