python ~/Megatron-DeepSpeed/tools/preprocess_data.py \
       --input ~/Megatron-DeepSpeed/tmp/example_train_0.jsonl \
       --output-prefix megatron_test \
	   --tokenizer-model ~/Megatron-DeepSpeed/tmp/tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
	   --split-sentences \
	   --workers 2
