import os
import requests
import sys

# AzureML libraries
import azureml.core
from azureml.core import Experiment, Workspace, Datastore, Run, Environment
from azureml.core.dataset import Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
#from azureml.contrib.core.compute.k8scompute import AksCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import ScriptRunConfig
from azureml.core.runconfig import PyTorchConfiguration, MpiConfiguration

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

ws = Workspace.from_config()
print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')

# Create the compute cluster
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

input_data_path = "https://megatrondsbookcorpus.blob.core.windows.net/bookcorpus/data"
input_dataset = Dataset.File.from_files(input_data_path)
# input_data_dir = input_dataset.as_mount()
input_data_dir = input_dataset.as_download()
# data_path = 'data/BookCorpus'
# vocab_file = 'data/gpt2-vocab.json'
# merge_file = 'data/gpt2-merges.txt'
tensorboard_dir = 'outputs/tensorboard'


# Create experiment
experiment_name = 'megatron-530b-ds-benchmark'
#experiment_name = 'megatron-175b-ds-benchmark-nvidia'
experiment = Experiment(ws, name=experiment_name)

megatron_ds_env = Environment.from_dockerfile(name='megatron-ds-ptca', dockerfile='Dockerfile.dockerfile')
#megatron_ds_env = Environment.from_dockerfile(name='megatron-ds-dockerfile-nvidia', dockerfile='Dockerfile.dockerfile')
#megatron_ds_env.register(ws).build(ws).wait_for_completion()

megatron_ds_env.environment_variables['NCCL_DEBUG'] = 'WARN'
#megatron_ds_env.environment_variables['NCCL_DEBUG_SUBSYS'] = 'INIT,GRAPH,ENV'
megatron_ds_env.environment_variables['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
megatron_ds_env.environment_variables['NCCL_SOCKET_IFNAME'] = 'eth0'
megatron_ds_env.environment_variables['NCCL_IB_PCI_RELAXED_ORDERING']='1'
megatron_ds_env.environment_variables['UCX_TLS']='tcp'
megatron_ds_env.environment_variables['UCX_NET_DEVICES']='eth0'
#megatron_ds_env.environment_variables['NCCL_TOPO_DUMP_FILE'] = 'topo.xml'
#megatron_ds_env.environment_variables['NCCL_ALGO'] = 'Tree'
#NCCL_DEBUG=INFO
#NCCL_DEBUG_SUBSYS=INIT,GRAPH
#NCCL_TOPO_DUMP_FILE=topo.xml
#NCCL_IB_PCI_RELAXED_ORDERING="1"
#CUDA_DEVICE_ORDER="PCI_BUS_ID"
#NCCL_SOCKET_IFNAME="eth0"

'''
NLAYERS=105
HIDDEN=20480
HEADS=160
SEQ=2048
            '--num-layers', 96, 
            '--hidden-size', 12288,
            '--num-attention-heads', 96, 
'''

run_args = ['--tensor-model-parallel-size', 1, 
            '--pipeline-model-parallel-size', 1, 
            '--num-layers', 105, 
            '--hidden-size', 20480,
            '--num-attention-heads', 160, 
            #'--num-layers', 10,
            #'--hidden-size', 2048,
            #'--num-attention-heads', 16, 
            '--seq-length', 1024,
            '--loss-scale', 15, 
            '--max-position-embeddings', 1024, 
            '--micro-batch-size', 4,
            '--global-batch-size', 1024,
            '--train-iters', 100,
            '--lr', 6.0e-5,
            '--min-lr', 6.0e-6, 
            '--lr-decay-style', 'cosine',
            '--log-interval', 1, 
            '--eval-iters', 40, 
            '--eval-interval', 1000,
            '--input-data-dir', input_data_dir,
            # '--data-path', data_path,
            # '--vocab-file', vocab_file,
            # '--merge-file', merge_file,
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
            # '--tensorboard-dir', str(output_ref),
            '--tensorboard-dir', tensorboard_dir,
            '--cpu-optimizer',
            '--remote-device', 'cpu'
            '--deepspeed',
            '--no-pipeline-parallel',
            '--deepspeed_config', 'ds_config.json',
            '--zero-stage', 3,
            '--deepspeed-activation-checkpointing',
            '--exit-interval', 5000
]

distr_config = PyTorchConfiguration(process_count=256, node_count=32)
#distr_config = MpiConfiguration(process_count_per_node=8, node_count=8)

megatron_ds_src = ScriptRunConfig(source_directory='../../',
                      script='pretrain_gpt.py',
                      arguments=run_args,
                      compute_target=compute_target,
                      environment=megatron_ds_env,
                      distributed_job_config=distr_config)


run = experiment.submit(megatron_ds_src, tags={'model':'530b', 'bs':'4', 'gpus':'256'})

#data_options=" \
#         --vocab-file ${VOCAB_PATH} \
#         --merge-file ${MERGE_PATH} \
#         --data-path ${DATA_PATH} \
#         --data-impl mmap"

# | tee ${OUTPUT_DIR}/output.log

#BASE_PATH=/shared/data/
#DATA_PATH=${BASE_PATH}/BookCorpusDataset_text_document
