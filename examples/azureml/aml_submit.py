import os
import requests
import sys

# AzureML libraries
import azureml.core
from azureml.core import Dataset, Environment, Experiment, ScriptRunConfig, Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import PyTorchConfiguration

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

# For setting up a workspace, refer to: https://github.com/Azure/azureml-examples/tree/main/python-sdk#set-up
ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

#-------------------------------------------------------------------------------
# Prepare Compute Cluster
#-------------------------------------------------------------------------------
cluster_name = "a100-80gb"

# Verify that the cluster doesn't exist already
try:
    compute_target = ComputeTarget(workspace=ws, name=cluster_name)
    print('Found existing compute target.')
except ComputeTargetException:
    print('Creating a new compute target...')
    compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_ND96amsr_A100_v4', min_nodes=32, max_nodes=32)
    
    # create the cluster
    compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

#-------------------------------------------------------------------------------
# Prepare Data
# Megatron-DeepSpeed takes in data_path, vocab_file, and merge_file.
# For AML, we are adding a parameter aml_data_download_path which specifies how to deliver the dataset to a compute target.
# In the submitted run, files in the datasets will be either mounted or downloaded to local path on the compute target.
# 
# data_path for this example is path to the .bin and .idx file, excluding extension.
# e.g. for data/BookCorpusDataset_text_document.bin and data/BookCorpusDataset_text_document.idx,
# data_path = "data/BookCorpusDataset_text_document"
#
# Once the folder is downloaded to the compute target, it will use aml_data_download_path to locate the folder
# and data_path to locate .bin and .idx files
#
# vocab_file and merge_file would also be passed in a similar way.
#-------------------------------------------------------------------------------
datastore = ws.get_default_datastore()
blobstore_datadir = "bookcorpus_data"
data_path = f"BookCorpusDataset_text_document"
# Load data folder which contains bookcorpus .bin and .idx files
train_dataset = Dataset.File.from_files(path=[(datastore, blobstore_datadir)])
aml_data_download_path = train_dataset.as_download(blobstore_datadir)

vocab_file_dataset = Dataset.File.from_files("https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json")
merge_file_dataset = Dataset.File.from_files("https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt")
vocab_file = vocab_file_dataset.as_download()
merge_file = merge_file_dataset.as_download()


#-------------------------------------------------------------------------------
# Setup training environment
#-------------------------------------------------------------------------------
megatron_ds_env = Environment.from_dockerfile(name='megatron-ds-ptca', dockerfile='Dockerfile.dockerfile')
megatron_ds_env.register(ws).build(ws).wait_for_completion()  # Comment this out if environment already exists

megatron_ds_env.environment_variables['NCCL_DEBUG'] = 'WARN'
megatron_ds_env.environment_variables['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
megatron_ds_env.environment_variables['NCCL_SOCKET_IFNAME'] = 'eth0'
megatron_ds_env.environment_variables['NCCL_IB_PCI_RELAXED_ORDERING']='1'
megatron_ds_env.environment_variables['UCX_TLS']='tcp'
megatron_ds_env.environment_variables['UCX_NET_DEVICES']='eth0'

#-------------------------------------------------------------------------------
# Training Settings and Arguments
#-------------------------------------------------------------------------------
node_count = 8
total_proceses_count = 64
micro_batch_size = 8
global_batch_size = micro_batch_size * total_proceses_count
tensorboard_dir = '/tmp/outputs/tensorboard'

run_args = ['--tensor-model-parallel-size', 1, 
            '--pipeline-model-parallel-size', 1, 
            '--num-layers', 96, 
            '--hidden-size', 12288,
            '--num-attention-heads', 96, 
            #'--num-layers', 10,
            #'--hidden-size', 2048,
            #'--num-attention-heads', 16, 
            '--seq-length', 1024,
            '--loss-scale', 15, 
            '--max-position-embeddings', 1024, 
            '--micro-batch-size', micro_batch_size,
            '--global-batch-size', global_batch_size,
            '--train-iters', 100,
            '--lr', 6.0e-5,
            '--min-lr', 6.0e-6, 
            '--lr-decay-style', 'cosine',
            '--log-interval', 1, 
            '--eval-iters', 40, 
            '--eval-interval', 1000,
            '--aml-data-download-path', aml_data_download_path,
            '--data-path', data_path,
            '--vocab-file', vocab_file,
            '--merge-file', merge_file,
            '--save-interval', 1000, 
            '--split', '98,2,0',
            '--clip-grad', 1.0, 
            '--weight-decay', 0.1,
            '--adam-beta1', 0.9,
            '--adam-beta2', 0.95,
            '--init-method-std', 0.006,
            '--fp16',
            '--data-impl', 'mmap',
            '--checkpoint-activations',
            '--tensorboard-dir', tensorboard_dir,
            #'--cpu-optimizer',
            '--deepspeed',
            '--no-pipeline-parallel',
            '--deepspeed_config', 'ds_config.json',
            '--zero-stage', 3,
            '--deepspeed-activation-checkpointing',
            '--exit-interval', 5000,
]

distr_config = PyTorchConfiguration(process_count=total_proceses_count, node_count=node_count)

megatron_ds_src = ScriptRunConfig(source_directory='../../',
                      script='pretrain_gpt.py',
                      arguments=run_args,
                      compute_target=compute_target,
                      environment=megatron_ds_env,
                      distributed_job_config=distr_config)

#-------------------------------------------------------------------------------
# Create experiment and submit
#-------------------------------------------------------------------------------
experiment_name = 'megatron-175b-ds-benchmark-ptca'
experiment = Experiment(ws, name=experiment_name)

run = experiment.submit(megatron_ds_src, tags={'bs':micro_batch_size, 'gpus':total_proceses_count})
