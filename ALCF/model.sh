#!/bin/bash --login
#┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
#┃                 GPT MODEL SETTINGS                    ┃
#┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ Model / Architecture settings                        ┃
# ┃ ---------------------------------------------------- ┃
# ┃ GPT-3 models use 2K sequence length/context window   ┃
# ┃ The "GPT-3 XXX" below are configs from GPT-3 paper   ┃
# ┃ https://arxiv.org/abs/2005.14165, choose based on    ┃
# ┃ your desired model size or build your own configs    ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3 Small:  125M ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="125M"
# NLAYERS=12
# HIDDEN=768
# ATEN_HEADS=12

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 1.5B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="1.5B"
# NLAYERS=48
# HIDDEN=1536
# ATEN_HEADS=24

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 1.5B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="1.5B"
# NLAYERS=48
# HIDDEN=1600
# ATEN_HEADS=25

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 2.7B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="2.7B"
# NLAYERS=32
# HIDDEN=2560
# ATEN_HEADS=32

# ┏━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 6.7B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="6.7B"
# NLAYERS=32
# HIDDEN=4096
# ATEN_HEADS=32

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 13B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="13B"
# NLAYERS=40
# HIDDEN=5120
# ATEN_HEADS=40

# ┏━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 18.4B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="18.4B"
# NLAYERS=40
# HIDDEN=6144
# ATEN_HEADS=48

# ┏━━━━━━━━━━━━━━━━━━━━━┓
# ┃ ✓ GPT-3: 20B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="20B"
# NLAYERS=44
# HIDDEN=6144
# ATEN_HEADS=64

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 25B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="25B"
# NLAYERS=64
# ------------
# HIDDEN=5760  # DEFAULT (no flash attn)
# ATEN_HEADS=64
# ------------
# HIDDEN=5888    # headdim = 5888 / 46 = 128
# ATEN_HEADS=46
# -----------------
# -- FLASH ATTN --
# headdim = 5760 / 80 = 72
# HIDDEN=5760
# ATEN_HEADS=80
# ------------

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 30B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="30B"
# NLAYERS=64
# HIDDEN=6144
# ATEN_HEADS=64

# ┏━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 33B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━┛
MODEL_SIZE="33B"
NLAYERS=80                                                                                                                                                                                                  
HIDDEN=5760                                                                                                                                                                                                      
ATEN_HEADS=80

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 145B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="145B"
# NLAYERS=80
# HIDDEN=12288
# ATEN_HEADS=96

# ┏━━━━━━━━━━━━━━━━━━━━┓
# ┃ GPT-3: 175B Params ┃
# ┗━━━━━━━━━━━━━━━━━━━━┛
# MODEL_SIZE="175B"
# NLAYERS=96
# HIDDEN=12288
# ATEN_HEADS=96


export MODEL_SIZE="${MODEL_SIZE}"
export NLAYERS="${NLAYERS}"
export HIDDEN="${HIDDEN}"
export ATEN_HEADS="${ATEN_HEADS}"
