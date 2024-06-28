# Usage : python mds_to_hf.py --mds_checkpoint <path/to/checkpoint/dir/mp_rank_*.pt> --output_dir <path/to/dir/to/store/hf/checkpoints>
# Tips : Do not run on login node. 
# This script currently only takes care of tp=1. Takes a AuroraGPT Llama model trained with Megatron-DeepSpeed and converts to LLamaCausalForLM architecture from HuggingFace. 

import argparse
import torch
import pdb
import os
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

def repeat_kv_wt(x,np):
    return torch.repeat_interleave(x, dim=0, repeats=np)

def Update_llama_config(Llama_config, mds_args):
    if mds_args['swiglu']:
        Llama_config.hidden_act = "silu"
    Llama_config.hidden_size = mds_args['hidden_size']
    Llama_config.intermediate_size = mds_args['ffn_hidden_size']
    Llama_config.max_position_embeddings = mds_args['max_position_embeddings']
    Llama_config.num_attention_heads = mds_args['num_attention_heads']
    Llama_config.num_hidden_layers = mds_args['num_layers']
    Llama_config.num_key_value_heads = mds_args['num_key_value_heads']
    Llama_config.rms_norm_eps = mds_args['layernorm_epsilon']
    Llama_config.rope_theta = mds_args['rope_theta']
    Llama_config.vocab_size = mds_args['padded_vocab_size']
    return Llama_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mds_checkpoint', required=True)   
    parser.add_argument('--output_dir', required=True)   
    args = parser.parse_args()

    # make output_dir if it does not exits.
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    filename = str(args.mds_checkpoint) 
    if not filename.split("/")[-1].startswith('mp_rank') and not filename.split("/")[-1].endswith('.pt'):
        assert ("Provide the right file path, The file should be of format mp_rank_*.pt")
    print(f"loading mds checkpoint {filename}")
 
    mds_model = torch.load(args.mds_checkpoint,map_location=torch.device('cpu'))
    Llama_model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",cache_dir='/eagle/datascience/vsastry/huggingface')
    
    Llama_config = Llama_model.config 
    Updated_Llama_config = Update_llama_config(Llama_config, mds_model['args'].__dict__)
    # save the updated config.json file 
    Updated_Llama_config.to_json_file(os.path.join(args.output_dir,'config.json'))

    state_dict = {}
    dim = mds_model['args'].__dict__['kv_channels']
    inv_freq = 1.0 / (mds_model['args'].__dict__['rope_theta'] ** (torch.arange(0,dim, 2).float() / dim))
    hidden_size = mds_model['args'].__dict__['hidden_size']
    kv_dim = mds_model['args'].__dict__['kv_channels'] * mds_model['args'].__dict__['num_key_value_heads']
    kv_groups = mds_model['args'].__dict__['num_attention_heads'] // mds_model['args'].__dict__['num_key_value_heads']
    for layer_i in range(Updated_Llama_config.__dict__['num_hidden_layers']):
        # SELF ATTENTION layers.
        # get the q, k, v weights separately. Keeping k and v at the GQA head dim, since the transformers/models/llama/modelling_utils will take care of it. 
        fused_qkv = mds_model['module']['language_model']['encoder'][f"layers.{layer_i}.self_attention.query_key_value.weight"]
        state_dict[f"model.layers.{layer_i}.self_attn.q_proj.weight"] = fused_qkv[0:hidden_size]
        state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = fused_qkv[hidden_size:hidden_size+kv_dim]
        #state_dict[f"model.layers.{layer_i}.self_attn.k_proj.weight"] = repeat_kv_wt(fused_qkv[hidden_size:hidden_size+kv_dim], kv_groups)
        state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = fused_qkv[hidden_size+kv_dim:hidden_size+2*kv_dim]
        #state_dict[f"model.layers.{layer_i}.self_attn.v_proj.weight"] = repeat_kv_wt(fused_qkv[hidden_size+kv_dim:hidden_size+2*kv_dim],kv_groups)
        state_dict[f"model.layers.{layer_i}.self_attn.o_proj.weight"] = mds_model['module']['language_model']['encoder'][f"layers.{layer_i}.self_attention.dense.weight"]
        
        # MLP Layers 
        fused_mlp = mds_model['module']['language_model']['encoder'][f"layers.{layer_i}.mlp.dense_h_to_4h.weight"]
        chunked_mlp = torch.chunk(fused_mlp,2,dim=0)
        state_dict[f"model.layers.{layer_i}.mlp.gate_proj.weight"] = chunked_mlp[0]
        state_dict[f"model.layers.{layer_i}.mlp.up_proj.weight"] = chunked_mlp[1]
        state_dict[f"model.layers.{layer_i}.mlp.down_proj.weight"] = mds_model['module']['language_model']['encoder'][f"layers.{layer_i}.mlp.dense_4h_to_h.weight"]
        
        #LayerNorm weights and RoPe 
        state_dict[f"model.layers.{layer_i}.input_layernorm.weight"] = mds_model['module']['language_model']['encoder'][f"layers.{layer_i}.input_layernorm.weight"]
        state_dict[f"model.layers.{layer_i}.post_attention_layernorm.weight"] = mds_model['module']['language_model']['encoder'][f"layers.{layer_i}.post_attention_layernorm.weight"] 

        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

    # Get the non-encoder layer weights. 
    state_dict["model.embed_tokens.weight"] = mds_model['module']['language_model']['embedding']['word_embeddings']['weight']
    state_dict["model.norm.weight"] = mds_model['module']['language_model']['encoder']['final_layernorm.weight']
    state_dict["lm_head.weight"] = mds_model['module']['language_model']['output_layer']['weight']
    
    # Save the model in the hf output path. 
    torch.save(state_dict, os.path.join(args.output_dir,"pytorch_model.bin"))    



