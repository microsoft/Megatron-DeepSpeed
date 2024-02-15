# Light-weight Library of Zero Bubble Pipeline Parallelism

## How to use
1. Insert the following code snippet to your training script at the very beginning:
```python
from light.zbpp_light.zbpp_light import patch_deepspeed
patch_deepspeed()

# Your original training script starts here
import megatron, etc
```

## Supported Frameworks
- [ ] Megatron-LM
- [x] Megatron-DeepSpeed

## Current Limitations
- Only supports ZB-H1 schedule, which reduces 2/3 1F1B bubble with same memory and communication cost.