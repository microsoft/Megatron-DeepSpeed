#!/usr/bin/env python
from megatron.data.gpt_dataset import build_train_valid_test_datasets
import numpy as np
from megatron.global_vars import set_args, set_global_variables, get_args
from megatron.arguments import parse_args 
from megatron.initialize import initialize_megatron
from megatron.data.data_samplers import build_pretraining_data_loader
from mpi4py import MPI
from megatron.core import mpu
comm = MPI.COMM_WORLD
initialize_megatron(allow_no_cuda=True)
args = get_args()

data_file_list = args.data_file_list
if comm.rank==0:
    print(f"Reading data from {args.data_file_list}")
files = []
weights = []
flist = []
with open(data_file_list, 'r') as fin:
    for f in fin.readlines():
        w, fname = f.split()
        weights.append(float(w))
        flist.append(fname)
        files.append(float(w))
        files.append(fname)
splits_string="100,0,0"

weights = np.array(weights)
weights = weights/np.sum(weights)

num_samples = args.global_batch_size*args.train_iters
num_datasets = len(weights)
if comm.rank==0:
    print(f"Number of datasets: {num_datasets}")
    print(f"Global batch size: {args.global_batch_size}")
    print(f"Training iterations: {args.train_iters}")
train_valid_test_num_samples = [num_samples, 0, 0]
seed=args.seed
data_impl = args.data_impl
skip_warmup = not args.mmap_warmup
seq_length = args.seq_length
splits_string = "1,0,0"

# Build datasets
train_ds, valid_ds, test_ds = build_train_valid_test_datasets(files, data_impl, splits_string,
                            train_valid_test_num_samples,
                            seq_length, seed, skip_warmup, data_cache_path=args.data_cache_path)

dataset_idx = [train_ds.dataset_index[i] for i in range(num_samples)]
ratio_select=np.zeros(num_datasets)
#for i in range(num_datasets):
#    ratio_select[i] = np.sum([i==d for d in dataset_idx])/num_samples
if comm.rank ==0:
    print(f"Total number of samples: {len(train_ds)}")
    print(f"Weights set: {weights[:min(8, num_datasets)]}")
#print(f"Weights across training: {ratio_select[:min(8, num_datasets)]}")

for e in range(min(100, args.train_iters)):
    ratio_select=np.zeros(num_datasets)
    for i in range(num_datasets):
        ratio_select[i] = np.sum([i==d for d in dataset_idx[e*args.global_batch_size:(e+1)*args.global_batch_size]])/args.global_batch_size
    if comm.rank==0:
        print(f"iter-{e}: {ratio_select[:min(8, num_datasets)]}")


print("First 10 samples")
for i in range(10):
    if comm.rank==0:
        print(f"Sample: {i} \t dataset_idx: {train_ds.dataset_index[i]}, sample_idx: {train_ds.dataset_sample_index[i]}")

#### Build data loaders
rank_in_parallel_group = mpu.get_sequence_parallel_rank()
print(rank_in_parallel_group)
if rank_in_parallel_group == 0:
    train_dataloader = build_pretraining_data_loader(
        train_ds, args.consumed_train_samples)
    valid_dataloader = build_pretraining_data_loader(
        valid_ds, args.consumed_valid_samples)
    test_dataloader = build_pretraining_data_loader(test_ds, 0)
