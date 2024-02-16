#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

OUTPUT_PATH=$1

if [ "$OUTPUT_PATH" == "" ]; then
    OUTPUT_PATH="z1_uni_ckpt"
fi

# Training Loss
python3 examples_deepspeed/universal_checkpointing/tb_analysis/tb_analysis.py \
    --tb_dir $OUTPUT_PATH \
    --tb_event_key "lm-loss-training/lm loss" \
    --plot_name "uc_char_training_loss.png" \
    --plot_title "Megatron-GPT Universal Checkpointing - Training Loss"

# Validation Loss
python3 examples_deepspeed/universal_checkpointing/tb_analysis/tb_analysis.py \
    --tb_dir $OUTPUT_PATH \
    --tb_event_key "lm-loss-validation/lm loss validation" \
    --csv_name "val" --plot_name "uc_char_validation_loss.png" \
    --plot_title "Megatron-GPT Universal Checkpointing - Validation Loss" \
    --plot_y_label "Validation LM Loss"
