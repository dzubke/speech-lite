# standard libraries
import argparse
import contextlib
import os
import itertools
import json
import logging
import math
from pathlib import Path
import random
import time
# third-party libraries
import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import tqdm
import yaml
# project libraries
import speech
import speech.loader as loader
from speech.models.ctc_model_train import CTC_train
from speech.utils.checkpoint import GCSCheckpointHandler
from speech.utils.io import load_config, load_from_trained, read_pickle, save, write_pickle 
from speech.utils.logging import get_logger, get_logger_filename
from speech.utils.model_debug import (
    check_nan_params_grads, log_batchnorm_mean_std, log_cpu_mem_disk_usage, log_model_grads,
    log_param_grad_norms, plot_grad_flow_line, plot_grad_flow_bar, save_batch_log_stats
)


def run_epoch(model, 
              optimizer, 
              train_ldr,
              logger, 
              debug_mode:bool, 
              tbX_writer, 
              iter_count:int, 
              avg_loss:float, 
              local_rank:int, 
              loss_name:str, 
              save_path:str,
              gcs_ckpt_handler,
              scaler:GradScaler=None)->tuple:
    """
    Performs a forwards and backward pass through the model
    Args:
        iter_count (int): count of iterations
        save_path (str): path to directory where model is saved
        gcs_ckpt_handler: facilities saving files to google cloud storage
        scaler (GradScaler): gradient scaler to prevent gradient underflow when autocast
            uses float16 precision for forward pass
    Returns:
        Tuple[int, float]: train state of # batch iterations and average loss
    """
    # booleans and constants for logging
    is_rank_0 = (torch.distributed.get_rank() == 0)
    use_log = (logger is not None and is_rank_0)
    log_modulus = 100     # limits certain logging function to report less frequently
    exp_w = 0.985        # exponential weight for exponential moving average loss        
    avg_grad_norm = 0
    model_t, data_t = 0.0, 0.0
    end_t = time.time()

    # progress bar for rank_0 process
    tq = tqdm.tqdm(train_ldr)  if is_rank_0 else train_ldr
    
    # counter for model checkpointing
    batch_counter = 0
    device = torch.device("cuda:" + str(local_rank))
    
    # if scaler is enabled, amp is being used
    use_amp = scaler.is_enabled()
    print(f"Amp is being used: {use_amp}")
    
    # training loop
    for batch in tq:
        if use_log: logger.info(f"train: ====== Iteration: {iter_count} in run_epoch =======")
        
        ##############  Mid-epoch checkpoint ###############
        if is_rank_0 \
        and batch_counter % (len(train_ldr) // gcs_ckpt_handler.chkpt_per_epoch) == 0 \
        and batch_counter != 0:
            preproc = train_ldr.dataset.preproc
            save(model.module, preproc, save_path, tag='ckpt')
            gcs_ckpt_handler.upload_to_gcs("ckpt_model_state_dict.pth")
            gcs_ckpt_handler.upload_to_gcs("ckpt_preproc.pyc")
            # save the run_sate
            ckpt_state_path = os.path.join(save_path, "ckpt_run_state.pickle")
            write_pickle(ckpt_state_path, {'run_state': (iter_count, avg_loss)})
            gcs_ckpt_handler.upload_to_gcs("ckpt_run_state.pickle")
            # checkpoint tensorboard
            gcs_ckpt_handler.upload_tensorboard_ckpt()    
        
        batch_counter += 1
        ####################################################

        # convert the temprorary generator batch to a permanent list
        batch = list(batch) 
        
        # save the batch information
        if use_log: 
            if debug_mode:  
                save_batch_log_stats(batch, logger)
                log_batchnorm_mean_std(model.module.state_dict(), logger)
 
        start_t = time.time()
        optimizer.zero_grad(set_to_none=True)   # set grads to None for modest perf improvement
        
        #  will autocast to lower precision if amp is used. otherwise, it's no-operation
        with autocast(enabled = use_amp):
            # unpack the batch 
            inputs, labels, input_lens, label_lens = model.module.collate(*batch)
            inputs = inputs.cuda() #.to(device) #.cuda(local_rank)
            out, rnn_args = model(inputs, softmax=False)
            
            # use the loss function defined in `loss_name`
            if loss_name == "native":
                loss = native_loss(out, labels, input_lens, label_lens, model.module.blank)
            elif loss_name == "awni":
                loss = awni_loss(out, labels, input_lens, label_lens, model.module.blank)
            elif loss_name == "naren":
                loss = naren_loss(out, labels, input_lens, label_lens, model.module.blank)
       
        # backward pass 
        loss = loss.cuda()      # amp needs the loss to be on cuda
        scaler.scale(loss).backward() 
        
        if use_log: 
            if debug_mode: 
                plot_grad_flow_bar(model.module.named_parameters(),  get_logger_filename(logger))
                log_param_grad_norms(model.module.named_parameters(), logger)

        # gradient clipping and optimizer step, scaling disabled if amp is not used
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 200).item()
        scaler.step(optimizer)
        scaler.update()

        # logging in rank_0 process
        if is_rank_0:
            # calculate timers
            prev_end_t = end_t
            end_t = time.time()
            model_t += end_t - start_t
            data_t += start_t - prev_end_t
            
            # creating scalers from grad_norm and loss for weighted
            # TODO, needed with pytorch 0.4, may not be necessary anymore
            if isinstance(grad_norm, torch.Tensor):
                grad_norm = grad_norm.item()
            if isinstance(loss, torch.Tensor):
                loss = loss.item()

            # calculating the weighted average of loss and grad_norm
            if iter_count == 0:
                avg_loss = loss
                avg_grad_norm = grad_norm
            else: 
                avg_loss = exp_w * avg_loss + (1 - exp_w) * loss
                avg_grad_norm = exp_w * avg_grad_norm + (1 - exp_w) * grad_norm
            
            # writing to the tensorboard log files
            tbX_writer.add_scalars('train/loss', {"loss": loss}, iter_count)
            tbX_writer.add_scalars('train/loss', {"avg_loss": avg_loss}, iter_count)
            
            # adding this to suppress a tbX WARNING about inf values
            # TODO, this may or may not be a good idea as it masks inf in tensorboard
            if grad_norm == float('inf') or math.isnan(grad_norm):
                tbX_grad_norm = 1
            else:
                tbX_grad_norm  = grad_norm
            tbX_writer.add_scalars('train/grad', {"grad_norm": tbX_grad_norm}, iter_count)

            # progress bar update    
            tq.set_postfix(
                it=iter_count, 
                grd_nrm=grad_norm,
                lss=loss, 
                lss_av=avg_loss, 
                t_mdl=model_t, 
                t_data=data_t,
                scl=scaler.get_scale())
            if use_log: 
                logger.info(f'train: loss is inf: {loss == float("inf")}')
                logger.info(
                    f"train: iter={iter_count}, loss={round(loss,3)}, grad_norm={round(grad_norm,3)}"
                )
        
            if iter_count % log_modulus == 0:
                if use_log: log_cpu_mem_disk_usage(logger)
        
        # checks for nan gradients
        if check_nan_params_grads(model.module.parameters()):
            print("\n~~~ NaN value detected in gradients or parameters ~~~\n")
            if use_log:
                logger.error(
                    f"train: labels: {[labels]}, label_lens: {label_lens} state_dict: {model.module.state_dict()}"
                )
                log_model_grads(model.module.named_parameters(), logger)
                save_batch_log_stats(batch, logger)
                log_param_grad_norms(model.module.named_parameters(), logger)
                plot_grad_flow_bar(model.module.named_parameters(), get_logger_filename(logger))
            
            #debug_mode = True
            #torch.autograd.set_detect_anomaly(True)

        iter_count += 1

    return iter_count, avg_loss


def native_loss(out, labels, input_lens, label_lens, blank_idx):
    """Calculates the loss using pytorch's native loss function. 
    Only works with pytorch 1.X. The log_softmax is performed outside of
    the loss function (unlike awni and naren's where the log_softmax is internal).
    The `.permute(1,0,2).float()` transform puts the log_probs in the expected format.
    """ 
    log_probs = nn.functional.log_softmax(out, dim=2) 
    loss_fn = torch.nn.CTCLoss(blank=blank_idx, reduction='sum', zero_infinity=True)
    loss = loss_fn(log_probs.permute(1,0,2).float(), labels, input_lens, label_lens)
    return loss

def awni_loss(out, labels, input_lens, label_lens, blank_idx):
    """Calculates the loss using awni hannun's warpctc bindings.
    Only works with pytorch 0.4.
    """
    import functions.ctc as ctc #awni hannun's ctc bindings
    loss_fn = ctc.CTCLoss(blank_label=blank_idx)   
    loss = loss_fn(out, labels, input_lens, label_lens)      
    return loss

def naren_loss(out, labels, input_lens, label_lens, blank_idx):
    """Calculates the loss function using sean naren's warpctc bindings.
    The `.permute(1,0,2).float().cpu()` section of the model output is meant
    to match the expected format for the loss function. the `.cpu()` call is necessary
    to calculate a non-zero loss value. 
    """
    from warpctc_pytorch import CTCLoss
    loss_fn = CTCLoss(blank=blank_idx, size_average=True, length_average=False)
    out = out.permute(1,0,2).float().cpu() #permuation for naren's  warpctc
    loss = loss_fn(out, labels, input_lens, label_lens)
    return loss


def eval_dev(model, ldr, preproc,  logger, loss_name):
    """
    Runs the devset evaluation loop.
    """
    losses = []; all_preds = []; all_labels = []
        
    model.set_eval()
    preproc.set_eval()  # turns off dataset augmentation
    use_log = (logger is not None)

    # saves time by not computing and saving gradients as there is no backwards pass
    with torch.no_grad():
        for batch in tqdm.tqdm(ldr):
            batch = list(batch)
            preds = model.infer(batch)
            
            inputs, labels, input_lens, label_lens = model.collate(*batch)
            inputs = inputs.cuda(non_blocking=True)
            out, rnn_args = model(inputs, softmax=False)

            if loss_name == "native":
                loss = native_loss(out, labels, input_lens, label_lens, model.blank)
            elif loss_name == "awni":
                loss = awni_loss(out, labels, input_lens, label_lens, model.blank)
            elif loss_name == "naren":
                loss = naren_loss(out, labels, input_lens, label_lens, model.blank)

            losses.append(loss.item())
            all_preds.extend(preds)
            all_labels.extend(batch[1])        #add the labels in the batch object
            
    loss = sum(losses) / len(losses)

    # decodes from integer tokens back to phoneme labels
    results = [(preproc.decode(l), preproc.decode(p))
               for l, p in zip(all_labels, all_preds)]
    
    cer = speech.compute_cer(results)
    print("Dev: Loss {:.3f}, CER {:.3f}".format(loss, cer))
    
    if use_log: 
        logger.info(f"eval_dev: loss calculated as: {loss.item():0.3f}")
        logger.info(f"eval_dev: loss is nan: {math.isnan(loss.item())}")
        logger.info(f"eval_dev: results {results}")
        logger.info(f"CER: {cer}")

    # set the model and preproc back to training mode
    model.set_train()
    preproc.set_train()

    return loss, cer


def run(local_rank:int, config:dict)->None:
    """Main function that defines the data, optimizer, and model objects and runs the training
    and evaluation loops.

    Args:
        local_rank (int): rank of the process on the GPU
        config (dict): training configuration dict
    """
    # unpacking the config
    data_cfg = config["data"]
    log_cfg = config["logger"]
    preproc_cfg = config["preproc"]
    opt_cfg = config["optimizer"]
    model_cfg = config["model"]
    train_cfg = config['training']  
    ckpt_cfg = config['checkpoint']  

    gcs_ckpt_handler = GCSCheckpointHandler(ckpt_cfg)
    
    # save the config to gcs
    os.make_dirs(ckpt_cfg['local_save_path'], exist_ok=True)
    with open(os.path.join(ckpt_cfg['local_save_path'], "ctc_config.yaml"), 'w') as fid:
        yaml.dump(config, fid)
    gcs_ckpt_handler.upload_to_gcs("ctc_config.yaml")
    
    # setting up the distributed training environment
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    print(f"local_rank: {local_rank}, dist.get_rank: {torch.distributed.get_rank()}")
    is_rank_0 = (torch.distributed.get_rank() == 0)

    # defining the logging and debugging modes
    use_log = log_cfg["use_log"] and is_rank_0
    debug_mode = log_cfg["debug_mode"]    
    if debug_mode: torch.autograd.set_detect_anomaly(True)

    # create a logger, rank_0 boolean is contained in `use_log`
    logger = get_logger("train_log", log_cfg['log_file'], log_cfg['level']) if use_log else None
   
    # creates tensorboardX writer in rank_0 process 
    tbX_writer = SummaryWriter(logdir=ckpt_cfg["local_save_path"]) if is_rank_0 else None

    
    # Load previous train state: dict with contents:
        # {start_epoch: int, run_state: (int, float), best_so_far: float, learning_rate: float}
    train_state_path = gcs_ckpt_handler.download_from_gcs_bucket(
        os.path.join(ckpt_cfg['gcs_dir'], "train_state.pickle")
    )
    if train_state_path:
        print(f"load train_state from: {train_state_path}")
        train_state = read_pickle(train_state_path)
    # if train_path doesn't exist, create empty dict to load from config 
    else:   
        print(f"load train_state from config")
        train_state = dict()
        
    # the get-statements will load from train_state if key exists, and from opt_cfg otherwise
    run_state = train_state.get('run_state',  opt_cfg['run_state'])
    best_so_far = train_state.get('best_so_far', opt_cfg['best_so_far'])
    start_epoch =  train_state.get('start_epoch', opt_cfg['start_epoch'])
    
    # create the loaders
    batch_size = opt_cfg["batch_size"]
    preproc = loader.Preprocessor(
        data_cfg["train_set"], 
        preproc_cfg, 
        logger, 
        start_and_end=data_cfg["start_and_end"]
    )
    
    train_ldr = loader.make_ddp_loader(data_cfg["train_set"], preproc, batch_size, num_workers=data_cfg["num_workers"])

    # create the dev-set loaders in the rank_0 process
    if is_rank_0:
        dev_ldr_dict = dict() 
        for dev_name, dev_path in data_cfg["dev_sets"].items():
            dev_ldr = loader.make_loader(dev_path, preproc, batch_size=8, num_workers=data_cfg["num_workers"])
            dev_ldr_dict.update({dev_name: dev_ldr})

    # Model
    model_cfg.update({'blank_idx': preproc_cfg['blank_idx']})   # add the blank_idx to model_cfg
    model = CTC_train(
        preproc.input_dim,
        preproc.vocab_size,
        model_cfg
    )
    if model_cfg["load_trained"]:
        local_trained_path = gcs_ckpt_handler.download_from_gcs_bucket(model_cfg['gcs_trained_path'])
        if not local_trained_path:
            print(f"no model found at gcs location: {model_cfg['gcs_trained_path']}")
        else:
            model_cfg['local_trained_path'] = local_trained_path
            model = load_from_trained(model, model_cfg)
            print(f"Succesfully loaded weights from trained model: {model_cfg['local_trained_path']}")
    
    # Optimizer and learning rate scheduler
    learning_rate = opt_cfg['learning_rate']
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,   # from train_state or opt_config
        momentum=opt_cfg["momentum"],
        dampening=opt_cfg["dampening"]
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=opt_cfg["sched_step"], 
        gamma=opt_cfg["sched_gamma"]
    )

    # gradient scaler, too large a value for init_scale produces NaN gradients
    scaler = GradScaler(enabled=train_cfg['amp'], init_scale=16)

    # call the ddp wrappers
    model.cuda(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    
    if use_log: 
        logger.info(f"train: ====== Model, loaders, optimimzer created =======")
        logger.info(f"train: model: {model}")
        logger.info(f"train: preproc: {preproc}")
        logger.info(f"train: optimizer: {optimizer}")
        logger.info(f"train: config: {config}")

    # printing to the output file
    if is_rank_0:
        print(f"====== Model, loaders, optimimzer created =======")
        print(f"model: {model}")
        print(f"preproc: {preproc}")
        print(f"optimizer: {optimizer}")
        print(f"config: {config}")


    # training loop
    for epoch in range(start_epoch, opt_cfg["epochs"]):
        
        start = time.time()
        for group in optimizer.param_groups:
            if is_rank_0: print(f'learning rate: {group["lr"]}')
            if use_log: logger.info(f"train: learning rate: {group['lr']}")
        
        try:
            run_state = run_epoch(
                model, optimizer, train_ldr, logger, debug_mode, tbX_writer, *run_state, local_rank,
                train_cfg['loss_name'], ckpt_cfg['local_save_path'], gcs_ckpt_handler, scaler
            )
        except Exception as err:
            if use_log: 
                logger.error(f"Exception raised: {err}")
                logger.error(f"train: ====In except block====")
                logger.error(f"train: state_dict: {model.module.state_dict()}")
                log_model_grads(model.module.named_parameters(), logger)
            raise Exception('Failure in run_epoch').with_traceback(err.__traceback__)
        finally: # used to ensure that plots are closed even if exception raised
            plt.close('all')
    
        # update the learning rate
        lr_scheduler.step()       
 
        if use_log:
            logger.info(f"train: ====== Run_state finished =======") 
            logger.info(f"train: preproc type: {type(preproc)}")
        if is_rank_0:
            msg = "Epoch {} completed in {:.2f} (hr)."
            epoch_time_hr = (time.time() - start)/60/60
            print(msg.format(epoch, epoch_time_hr))
            if use_log: logger.info(msg.format(epoch, epoch_time_hr))
            tbX_writer.add_scalars('train/stats', {"epoch_time_hr": epoch_time_hr}, epoch)
    
            # the logger needs to be removed to save the model
            if use_log: preproc.logger = None
            speech.save(model.module, preproc, ckpt_cfg["local_save_path"])
            gcs_ckpt_handler.upload_to_gcs("model_state_dict.pth")
            gcs_ckpt_handler.upload_to_gcs("preproc.pyc")

            if use_log: 
                logger.info(f"train: ====== model saved =======")
                preproc.logger = logger

            # creating the dictionaries that hold the PER and loss values
            dev_loss_dict = dict()
            dev_per_dict = dict()
            # iterating through the dev-set loaders to calculate the PER/loss
            for dev_name, dev_ldr in dev_ldr_dict.items():
                print(f"evaluating devset: {dev_name}")
                if use_log: logger.info(f"train: === evaluating devset: {dev_name} ==")
                dev_loss, dev_per = eval_dev(model.module, dev_ldr, preproc, logger, train_cfg['loss_name'])

                dev_loss_dict.update({dev_name: dev_loss})
                dev_per_dict.update({dev_name: dev_per})

                if use_log: logger.info(f"train: ====== eval_dev {dev_name} finished =======")
                
                # Save the best model on the dev set
                if dev_name == data_cfg['dev_set_save_reference']:
                    print(f"dev_reference {dev_name}: current PER: {dev_per} vs. best_so_far: {best_so_far}")
                    
                    if use_log: logger.info(f"dev_reference {dev_name}: current PER: {dev_per} vs. best_so_far: {best_so_far}")
                    if dev_per < best_so_far:
                        if use_log: preproc.logger = None   # remove the logger to save the model
                        best_so_far = dev_per
                        speech.save(model.module, preproc, ckpt_cfg["local_save_path"], tag="best")
                        gcs_ckpt_handler.upload_to_gcs("best_model_state_dict.pth")
                        gcs_ckpt_handler.upload_to_gcs("best_preproc.pyc")

                        if use_log: 
                            preproc.logger = logger
                            logger.info(f"model saved based per on: {dev_name} dataset")

                        print(f"UPDATED: best_model based on PER {best_so_far} for {dev_name} devset")
                
            per_diff_dict = calc_per_difference(dev_per_dict) 

            tbX_writer.add_scalars('dev/loss', dev_loss_dict, epoch)
            tbX_writer.add_scalars('dev/per', dev_per_dict, epoch)
            tbX_writer.add_scalars('dev/per/diff', per_diff_dict, epoch)
            gcs_ckpt_handler.upload_tensorboard_ckpt()  

            learning_rate = list(optimizer.param_groups)[0]["lr"]
            # save the current state of training
            train_state = {"start_epoch": epoch + 1, 
                           "run_state": run_state, 
                           "best_so_far": best_so_far,
                           "learning_rate": learning_rate}
            write_pickle(os.path.join(ckpt_cfg["local_save_path"], "train_state.pickle"), train_state)
            gcs_ckpt_handler.upload_to_gcs("train_state.pickle")


def calc_per_difference(dev_per_dict:dict) -> dict:
    """
    Calculates the differecence between the speak testset PER and the training-dev sets. This
    difference is a measure of data mismatch.
    """
    per_diff_dict = dict()

    for name, per in dev_per_dict.items():
        if not name=='speak':
            diff_name = name + "-speak"
            per_diff_dict[diff_name] = dev_per_dict.get('speak', 0.0) - dev_per_dict.get(name, 0.0)
    
    return per_diff_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Train a speech model.")
    parser.add_argument("config",
        help="A json file with the training configuration.")
    parser.add_argument("--deterministic", default=False,
        action="store_true",
        help="Run in deterministic mode (no cudnn). Only works on GPU.")
    #parser.add_argument('--rank', default=0, type=int,
    #                    help='ranking within the compute nodes')
    parser.add_argument('--local_rank', default=0, type=int,
                        help='local rank for singe node, aka gpu index.')
    args = parser.parse_args()

    config = load_config(args.config)

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    use_cuda = torch.cuda.is_available()

    if use_cuda and args.deterministic:
        torch.backends.cudnn.enabled = False

    train_cfg = config['training']    

    os.environ['OMP_NUM_THREADS'] = str(train_cfg['OMP_NUM_THREADS'])

    run(local_rank=args.local_rank, config=config)
