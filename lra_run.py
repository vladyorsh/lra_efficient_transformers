from lra.models import *
from lra.setups import *
from lra.utils  import *

import sys
import os
import argparse
import torch
import tensorflow as tf

#TODO:
#Check the static graph option
#Manage inputs without explicit keys
#Model optional args
#Use tfds API before making a Torch wrapper for better batching
#----------
#0.5 reg weight for matching
#Get rid of setup dictionaries
#Redo with Lightning CLI

def get_lra_data(lib_path, data_path, task, batch_size, max_length):
    sys.path.append(os.path.realpath(lib_path))
        
    from lra_benchmarks.matching.input_pipeline import get_matching_datasets
    from lra_benchmarks.listops.input_pipeline import get_datasets as get_listops_datasets
    from lra_benchmarks.text_classification.input_pipeline import get_tc_datasets
    
    MATCHING_DATADIR = os.path.join(os.path.realpath(data_path), 'lra_release/tsv_data')
    LISTOPS_DATADIR  = os.path.join(os.path.realpath(data_path), 'listops-1000/')
    
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
        
def get_model(args, encoder, setup):
    BASE_MODELS = { 'classification' : ClassificationTransformer, 'matching' : MatchingTransformer }
    LUNA_MODELS = { 'classification' : LunaClassifier,            'matching' : LunaMatcher }
    PRELUNA_MODELS = { 'classification' : PreLunaClassifier,      'matching' : PreLunaMatcher }
    SELFLUNA_MODELS= { 'classification' : SelfLunaClassifier,     'matching' : SelfLunaMatcher }
    
    REGISTERED_MODELS = {
        'base' : BASE_MODELS,
        'luna' : LUNA_MODELS,
        'preluna' : PRELUNA_MODELS,
        'selfluna' : SELFLUNA_MODELS,
    }
    
    ADDITIONAL_MODEL_ARGS = {
        'luna'      : ['mem_size'],
        'preluna'   : ['mem_size'],
        'selfluna'  : ['mem_size'],
    }
    
    task = 'classification' if args.task in { 'classification', 'listops' } else 'matching'
    model_class = REGISTERED_MODELS[args.model][task]
    additional_args = ADDITIONAL_MODEL_ARGS[args.model]
    additional_args = { arg : vars(args).get(arg) for arg in additional_args }
    model = LraLightningWrapper(
        model_class(
            classes=setup['classes'],
            num_embeddings=encoder.vocab_size,
            seq_len=args.max_length,
            hidden_dim=setup['hidden_dim'],
            qkv_dim=setup['qkv_dim'],
            mlp_dim=setup['mlp_dim'],
            num_heads=setup['num_heads'],
            num_blocks=setup['num_blocks'],
            internal_dropout_rate=setup['internal_dropout_rate'],
            output_dropout_rate=setup['output_dropout_rate'],
            affine=args.biases,
            logging_frequency=args.logging_frequency,
            ** additional_args
        ),
        reg_weight=1.0,
        betas=(0.9, 0.999), #Original LRA uses 0.98, but may yield quite unsatisfying results
        base_lr=setup['lr'],
        wd=setup['weight_decay'],
        schedule=setup['schedule'](),
        log_non_scalars=args.log_non_scalars,
        log_params=args.log_params,
        mask_inputs=args.mask_inputs,
    )
    
    return model
    
def get_batch_size_and_acc_steps(effective_batch_size, per_device_batch_size, devices, accelerator, strategy):
    
    if effective_batch_size % devices:
        raise ValueError('The SETUP BATCH SIZE is not divisible by the DEVICE COUNT for the chosen strategy!')
    if 'ddp' in strategy: #Other strategies may split the batch automatically between devices (be careful!)
        sampled_batch_size = per_device_batch_size * devices
        
        #The following is needed to avoid the CUDA_ERROR_OUT_OF_MEMORY on tf dataset processing
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else: #For ddp we sample a full batch
        sampled_batch_size = per_device_batch_size
    if effective_batch_size % sampled_batch_size:
        raise ValueError('The SETUP BATCH SIZE is not divisible by the EFFECTIVE ONE-PASS BATCH SIZE, try to select another per-device batch size!')
    if strategy == 'single':
        strategy = pl.strategies.SingleDeviceStrategy(device=0)
    if strategy == 'ddp_static':
        strategy = pl.strategies.DDPStrategy(static_graph=True)
    accumulation_steps = max(1, effective_batch_size // sampled_batch_size)
    return sampled_batch_size, accumulation_steps, strategy

def print_device_info():
    device_count = 0
    for device in range(torch.cuda.device_count()):
      device_count += 1

      t = torch.cuda.get_device_properties(device).total_memory
      r = torch.cuda.memory_reserved(device)
      a = torch.cuda.memory_allocated(device)
      t = t / 1024 ** 2 #MB

      device = torch.cuda.get_device_name(device)

      print(f'Device: {device}, memory reserved: {r}, memory allocated: {a}, memory total: {t}')

def main(args):
    setup = get_setup(args.task)
    print_device_info()
    
    #Parse the training strategy and determine the sizes of sampled batches
    sampled_batch_size, accumulation_steps, strategy = get_batch_size_and_acc_steps(setup['full_batch_size'], args.batch_size, args.devices, args.accelerator, args.strategy)
    
    print(f'Sampling {sampled_batch_size} samples according to the strategy, and applying {accumulation_steps} grad accumulation steps.')
    
    train_dataset, valid_dataset, test_dataset, encoder = get_lra_data(args.lib_path, args.data_path, args.task, args.batch_size, args.max_length)
    train_dataset, valid_dataset, test_dataset = wrap_lra_tf_dataset(train_dataset, num_workers=args.data_workers), wrap_lra_tf_dataset(valid_dataset, num_workers=args.data_workers), wrap_lra_tf_dataset(test_dataset, num_workers=args.data_workers)
    
    torch.set_float32_matmul_precision(args.matmul_precision)
    model = get_model(args, encoder, setup)
    print(model)
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        strategy=strategy,
        devices=args.devices,
        num_nodes=1,
        precision=args.precision,
        logger=pl.loggers.TensorBoardLogger('logs', name=args.exp_name), #pl.loggers.CSVLogger("logs", name=args.exp_name),
        callbacks=[
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
            #pl.callbacks.DeviceStatsMonitor(),
            #pl.callbacks.EarlyStopping(...),
            
            #Progress bars
            #pl.callbacks.RichProgressBar(refresh_rate=1, leave=True),
            #PBar(refresh_rate=50),
            pl.callbacks.TQDMProgressBar(refresh_rate=100),
            
            #Checkpointing
            pl.callbacks.ModelCheckpoint(monitor='val_accuracy', verbose=True, save_weights_only=False, mode='max', auto_insert_metric_name=True, every_n_train_steps=setup['eval_period'], save_on_train_epoch_end=False),
            
            #Early stopping
            pl.callbacks.EarlyStopping('val_accuracy', min_delta=0.0, patience=setup['patience'], verbose=True, mode='max', check_on_train_epoch_end=False),
            LunaStopperCallback(threshold_acc= 1/model.model.classes + 0.01, min_evaluations=10),
        ],
        max_steps=setup['steps'],
        check_val_every_n_epoch=None,
        val_check_interval=setup['eval_period'],
        accumulate_grad_batches=accumulation_steps,
        #!!!!!!!!
        fast_dev_run=args.fast,
        barebones=False,
    )
    trainer.fit(model, train_dataloaders=train_dataset, val_dataloaders=valid_dataset)
    trainer.test(model, dataloaders=test_dataset, ckpt_path='best', verbose=True)
    
if __name__ == "__main__":
    def bool_type(x):
        if x.lower() == 'true':
            return True
        elif x.lower() == 'false':
            return False
        raise ValueError(f'Cannot parse {x} as bool!')
        
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
    parser.add_argument('--exp_name', help='experiment name', default='my_exp_name')
    parser.add_argument('--log_non_scalars', help='log non-scalar artifacts, e.g. images', type=bool_type, default=True)
    parser.add_argument('--logging_frequency', help='log non-scalars every N steps', type=int, default=100)
    parser.add_argument('--matmul_precision', help='torch matmul precision ( medium | high | highest )', default='highest')
    parser.add_argument('--log_params', help='log model parameter histograms and weight pictures', type=bool_type, default=False)
    parser.add_argument('--mask_inputs', help='mask input [PAD] tokens', type=bool_type, default=False)
    parser.add_argument('--biases', help='enable biases and affine transforms', type=bool_type, default=True)
    parser.add_argument('--fast', help='fast dev run for debugging', type=bool_type, default=False)
    parser.add_argument('--mem_size', help='memory-augmented models memory size', type=int, default=256)
    parser.add_argument('--num_repeats', help='how many times to repeat the experiment', type=int, default=1)
    args = parser.parse_args()
    for i in range(args.num_repeats):
        print(f'Starting experiment iteration {i}...')
        main(args)