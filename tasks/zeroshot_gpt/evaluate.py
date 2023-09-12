# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""GPT zero-shot evaluation."""

import math

import torch

from megatron import get_args
from megatron import print_rank_0, is_last_rank
from megatron import get_tokenizer
from megatron.core import parallel_state, tensor_parallel, mpu
from megatron.checkpointing import load_checkpoint
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
from tasks.finetune_utils import build_data_loader
from deepspeed.accelerator import get_accelerator
from .datasets import build_dataset
from megatron.model.rotary_pos_embedding import apply_rotary_pos_emb, RotaryEmbedding

from megatron.optimizer import get_megatron_optimizer

# These are needed to unwrap the model, would be nice to put these in megatron.utils if possible?
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator
from deepspeed.runtime.config import DeepSpeedConfig

import json

def get_model_provider(eval_metric):
    """Based on evaluation metric set the parallel-output flag and
    return the model provider."""

    def model_provider(pre_process=True, post_process=True):
        """Build the model."""

        config = core_transformer_config_from_args(get_args())

        if eval_metric == 'loss':
            parallel_output = True
        elif eval_metric == 'accuracy':
            parallel_output = False
        else:
            raise NotImplementedError('output type for {} evaluation metric '
                                      'is not supported.'.format(eval_metric))

        print_rank_0('building GPT model ...')
        
        args = get_args()
        config = core_transformer_config_from_args(args)
        if args.deepspeed:
            with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                                    remote_device=None if args.remote_device == 'none' else args.remote_device,
                                    config_dict_or_path=args.deepspeed_config,
                                    enabled=args.zero_stage == 3,
                                    mpu=mpu):
            
                model = GPTModel(
                    config=config,
                    num_tokentypes=0,
                    parallel_output=True,
                    pre_process=pre_process,
                    post_process=post_process
                )
        else:
            model = GPTModel(config=config, num_tokentypes=0, parallel_output=parallel_output,
                         pre_process=pre_process, post_process=post_process)
                

        return model

    return model_provider


def process_batch(batch):
    """Process batch and produce inputs for the model."""
    args = get_args()
    tokenizer = get_tokenizer()

    loss_mask = batch['pad_mask'].long().to(get_accelerator().device_name()).contiguous().byte()
    tokens_ = batch['text'].long().to(get_accelerator().device_name()).contiguous()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, attention_mask, position_ids, loss_mask


def forward_step(batch, model, eval_metric):
    """Forward step."""

    # Get the batch.
    tokens, labels, attention_mask, position_ids, loss_mask = process_batch(
        batch)

    # Tell the model what our actual batch size will be
    args = get_args()
    args.micro_batch_size = len(labels)

    input_tensor = recv_forward()

    # Forward pass through the model.
    if not args.deepspeed:
        unwrapped_model = unwrap_model(
            model, (torchDDP, LocalDDP, Float16Module))
        unwrapped_model.set_input_tensor(input_tensor)
    output = model(tokens, position_ids, attention_mask)

    send_forward(output)

    if parallel_state.is_pipeline_last_stage():
        # For loss, return the unreduced loss.
        if eval_metric == 'loss':
            losses = tensor_parallel.vocab_parallel_cross_entropy(
                output[0].contiguous().float(), labels.contiguous())
            loss = torch.sum(
                losses.view(-1) * loss_mask.contiguous().view(-1).float())
            return loss

        # For accuracy, return the number of correctly predicted samples.
        if eval_metric == 'accuracy':
            outputs = torch.argmax(output, -1)
            correct = (outputs == labels).float()
            correct[(1 - loss_mask).bool()] = 1
            correct = correct.prod(-1)
            return correct.sum()

        raise NotImplementedError('forward method for evaluation metric {} '
                                  'is not implemented.'.format(eval_metric))
    return None


def evaluate(data_loader, model, eval_metric):
    """Evaluation."""
    args = get_args()

    # Turn on evaluation mode which disables dropout.
    model.eval()

    total_output = 0.0
    with torch.no_grad():
        # For all the batches in the dataset.
        for iteration, batch in enumerate(data_loader):
            if iteration % args.log_interval == 0:
                print_rank_0('> working on iteration: {}'.format(iteration))
            # Forward evaluation.
            output = forward_step(batch, model, eval_metric)

            # Reduce across processes.
            if parallel_state.is_pipeline_last_stage():
                torch.distributed.all_reduce(output,
                                             group=parallel_state.get_data_parallel_group())

                total_output += output

    return total_output


def evaluate_and_print_results(task, data_loader, model, eval_metric):
    """Evaluate and print results on screen."""

    # Evaluate and get results.
    output = evaluate(data_loader, model, eval_metric)

    string = ' validation results on {} | '.format(task)
    if is_last_rank():
        if eval_metric == 'loss':
            num_tokenized_tokens = data_loader.dataset.num_tokenized_tokens
            num_original_tokens = data_loader.dataset.num_original_tokens
            val_loss = output / (num_tokenized_tokens - 1)
            ppl = math.exp(min(20, val_loss))
            token_ratio = (num_tokenized_tokens - 1) / (num_original_tokens - 1)
            adjusted_ppl = math.exp(min(20, val_loss * token_ratio))
            string += 'avg loss: {:.4E} | '.format(val_loss)
            string += 'ppl: {:.4E} | '.format(ppl)
            string += 'adjusted ppl: {:.4E} | '.format(adjusted_ppl)
            string += 'token ratio: {} |'.format(token_ratio)

            results = {
                "loss": val_loss.item(),
                "ppl": ppl,
                "ajusted_ppl": adjusted_ppl,
                "token_ratio": token_ratio
            }

            with open('./eval_results', 'w') as json_file:
                json.dump(results, json_file)

        elif eval_metric == 'accuracy':
            num_examples = len(data_loader.dataset)
            acc = output / num_examples
            string += 'number correct: {:.4E} | '.format(output)
            string += 'total examples: {:.4E} | '.format(num_examples)
            string += 'avg accuracy: {:.4E}'.format(acc)

        else:
            raise NotImplementedError('evaluation method for {} metric is not '
                                      'implemented yet.'.format(eval_metric))

        length = len(string) + 1
        print('-' * length)
        print(string)
        print('-' * length)

def main():
    """Main program."""
    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    if args.task == 'LAMBADA':
        eval_metric = 'accuracy'
    elif args.task == 'WIKITEXT103':
        eval_metric = 'loss'
    else:
        raise NotImplementedError('{} task is not implemented.'.format(
            args.task))

    # Set up model and load checkpoint.
    model = get_model(get_model_provider(eval_metric), wrap_with_ddp=False)

    if args.deepspeed:
        optimizer = None
        opt_param_scheduler = None
        model, optimizer, _, opt_param_scheduler = deepspeed.initialize(
                model=model[0],
                model_parameters=model[0].parameters(),
                optimizer=optimizer,
                args=args,
                lr_scheduler=opt_param_scheduler,
                mpu=mpu if args.no_pipeline_parallel else None
            )
        model = [model]
    
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # Data stuff.
    dataset = build_dataset(args.task)
    dataloader = build_data_loader(dataset, args.micro_batch_size,
                                   args.num_workers, drop_last=False)

    # Run evaluation.
    evaluate_and_print_results(args.task, dataloader, model, eval_metric)
    

    print_rank_0('done :-)')

