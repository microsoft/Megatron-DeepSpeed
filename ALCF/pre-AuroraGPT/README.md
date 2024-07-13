# Megatron-DeepSpeed (@ [ALCF](https://alcf.anl.gov))

![image](https://github.com/argonne-lcf/Megatron-DeepSpeed/assets/5234251/f06df155-30e8-4894-a4c2-c17ff4b34ada)

We describe below the instructions for launching distributed training with
Microsoft's Megatron-DeepSpeed and briefly describe some parallelism
strategies and various optimizations that are supported.

> [!IMPORTANT]
> We maintain this (forked) version at
> [`argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)
> that has some [helper scripts](#helper-scripts) for launching and setting
> various training options.
> 
> These changes are entirely self-contained **HERE** in [`ALCF/`](.)

## Setup

1. Load `conda` and activate base environment:

    ```bash
    # load conda + activate base env
    module load conda/2023-10-04 ; conda activate base
    ```

1. Clone
   [`argonne-lcf/Megatron-DeepSpeed`](https://github.com/argonne-lcf/Megatron-DeepSpeed)
   and navigate into it:

    ```bash
    # clone + navigate into Megatron-DeepSpeed repo
    git clone https://github.com/argonne-lcf/Megatron-DeepSpeed
    cd Megatron-DeepSpeed
    ```

1. Make virtual environment (on top of base conda):

    ```bash
    # make virtual environment (on top of base conda)
    mkdir -p venvs/polaris/2023-10-04
    python3 -m venv venvs/polaris/2023-10-04 --system-site-packages
    source venvs/polaris/2023-10-04/bin/activate
    ```

1. Install missing dependency:

    ```bash
    # install *missing dependency
    python3 -m pip install "git+https://github.com/saforem2/ezpz"
    ```

1. Launch training:

    ```bash
    # ---- launch training -----------------------
    # - MODEL_SIZE_KEY: defined in ALCF/model.sh
    # - other args: defined in ALCF/args.sh
    # ---------------------------------------------
    MODEL_SIZE_KEY="GPT25B" \
        SEQ_LEN=4096 \
        USE_FLASH_ATTN_V2=1 \
        MICRO_BATCH=1 \
        GAS=1 \
        SP_TYPE="megatron" \
        ZERO_STAGE=1 \
        ./ALCF/train-gpt3.sh
    ```


## Helper Scripts

- [`pretrain_gpt_alcf.py`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/pretrain_gpt_alcf.py)
- 📂 [`ALCF/`](https://github.com/argonne-lcf/Megatron-DeepSpeed/tree/main/ALCF)  
  `├──` [`args.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/models.sh)  
  `├──` [`launch.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/launch.sh)  
  `├──` [`model.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/model.sh)  
  `├──` [`setup.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/setup.sh)  
  `├──` [`submit-pbs.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/submit-pbs.sh)  
  `├──` [`submit.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/submit.sh)  
  `└──` [`train-gpt3.sh`](https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/train-gpt3.sh)  


<dl>
  <dt><a href=https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/pretrain_gpt_alcf.py><code>pretrain_gpt_alcf.py</code></a>
  <dd>Python module to be launched. Running `./ALCF/train-gpt3.sh` will automaticall build an `mpiexec` command and launch this module.</dd>
  <dt><a href=https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/train-gpt3.sh><code>ALCF/train-gpt3.sh</code></a>
  <dd>Main entry point for training. This script will automatically source the rest of the required ALCF/*.sh scripts below</dd>
  <dt><a href=https://github.com/saforem2/Megatron-DeepSpeed/blob/main/ALCF/model.sh><code>ALCF/model.sh</code></a></dt>
  <dd>Contains some example model architectures for GPT3-style models</dd>
  <dt><a href="https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/args.sh"><code>ALCF/args.sh</code></a></dt>
  <dd>Logic for parsing / setting up runtime options for Megatron and DeepSpeed.</dd>
  <dt><a href="https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/setup.sh"><code>ALCF/setup.sh</code></a></dt>
  <dd>Locate and activate virtual environment to be used, ensure MPI variables are set properly</dd>
  <dt><a href="https://github.com/argonne-lcf/Megatron-DeepSpeed/blob/main/ALCF/launch.sh"><code>ALCF/launch.sh</code></a></dt>
  <dd>Identify available resources and build the command to be ran i.e. figure out how many: `{nodes, GPUs per node, GPUs total}`, to pass to `mpi{run,exec}` then, use this to build  `mpiexec {mpiexec-args} python3 pretrain_gpt.py`</dd>
</dl>
