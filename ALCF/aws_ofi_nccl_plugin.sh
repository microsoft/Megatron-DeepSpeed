#!/bin/bash --login

# AWS NCCL OFI Plugin settings below
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=1
export NCCL_NET="AWS Libfabric"
export LD_LIBRARY_PATH=/soft/libraries/aws-ofi-nccl/v1.9.1-aws/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/libraries/hwloc/lib/:$LD_LIBRARY_PATH
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_MR_CACHE_MONITOR=userfaultfd
export FI_CXI_DEFAULT_CQ_SIZE=131072
#########################################################
# WARNING: !!!
# - Currently, `export NCCL_NET_GDR_LEVEL=PHB`
#   causes a hang on Polaris.
#   so, we don't set it for the time being [2024-05-14].
# - Seems to work on Perlmutter ???
#
# export NCCL_NET_GDR_LEVEL=PHB
#########################################################
