# Adding memory profiling
change ./megatron/training.py by adding the following in ```def train()``` and ```def evaluate()```.
```
...
from contextlib import ExitStack
...
def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func):
    ...
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    with ExitStack() as stack:
        #if args.profile_execution and torch.distributed.get_rank() == 0:
        # if args.profile_execution:
        prof = stack.enter_context(
            torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./output/tensorboard/run_"
                    # + args.profile_name
                    +'_'+timestamp
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
        )
    while iteration < args.train_iters and (args.train_tokens is None or \
        args.consumed_train_tokens < args.train_tokens):
        ...
        prof.step()
...
def evaluate(forward_step_func,
             data_iterator,
             model,
             process_non_loss_data_func,
             config,
             verbose=False):
    ...
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    with ExitStack() as stack:
        prof = stack.enter_context(
            torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=2, warmup=1, active=2, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    "./output/tensorboard/run_inference"
                    # + args.profile_name
                    +'_'+timestamp
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            )
        )
    with torch.no_grad():
        iteration = 0
        while iteration < args.eval_iters:
            ...
            prof.step()
```
# Fixing Issues from the original repo
change ./megatron/core/pipeline_parallel/schedules.py for the following by adding ```and not args.inference```
```
...
def forward_backward_no_pipelining():
    if args.deepspeed and not args.inference:
        model.set_gradient_accumulation_boundary(False)
...
    if args.deepspeed and not args.inference:
        model.set_gradient_accumulation_boundary(True)
```
# Add profile names
./megatron/arguments.py
```
    group.add_argument('--profile-execution', type=bool, default=False,
                       help='Use pytorch profiler during execution ')
    group.add_argument('--profile-name', default=False,  help=' Profile folder name ')
```
# Add dataset files
./dataset
```
gpt2-merges.txt
gpt2-vocab.json
my-gpt2_text_document.bin
my-gpt2_text_document.idx
```
# Add experiment scripts
./examples_deepspeed/MoE/moe_inference.sh
# Run
```
bash ./examples_deepspeed/MoE/moe_inference.sh
```