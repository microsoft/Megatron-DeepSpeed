import sys

def patch_deepspeed():
    assert all([not x.startswith('deepspeed') for x in sys.modules.keys()]), 'Please patch zbpp before importing any deepspeed modules.'
    import deepspeed.runtime.pipe
    from deepspeed.runtime.pipe import schedule
    
    from light.zbpp_light.deepspeed_schedule import BackwardOnlyPass, WeightPass, ZeroBubble1POptimized
    from light.zbpp_light.deepspeed_engine import _exec_backward_only_pass, _exec_weight_pass
    

    deepspeed.runtime.pipe.schedule.TrainSchedule = ZeroBubble1POptimized
    deepspeed.runtime.pipe.engine.PipelineEngine._INSTRUCTION_MAP.update(
        {
            BackwardOnlyPass: _exec_backward_only_pass,
            WeightPass: _exec_weight_pass,
        }
    )