
from pathlib import Path
import datetime
import time

from azureml.core import Workspace
from azure.ml.component import (
    Component,
    dsl,
)

from azureml.core import Workspace, Dataset, Datastore
from azureml.data import OutputFileDatasetConfig
import itertools
from azureml.core.authentication import InteractiveLoginAuthentication


def main():
    # subscription_id = '9ec1d932-0f3f-486c-acc6-e7d78b358f9b'
    # resource_group = 'CopilotEyesOff'
    # workspace_name = 'copilotEyesOffWestUS3'
    # default_compute_target = "A100-40G-non-ipp"
    # ws = Workspace(subscription_id, resource_group, workspace_name)
    # ds = Datastore.get(ws, "blob_data_babela100")
    # train_dataset = Dataset.get_by_name(ws, "megatron_github_300B")
    # vocab_dataset = Dataset.get_by_name(ws, "megatron-lm-vocab")
    # train_dataset = Dataset.get_by_name(ws, "megatron_github_test")

    
    # subscription_id = '79f57c16-00fe-48da-87d4-5192e86cd047'
    # resource_group = 'alexanderopenai64'
    # workspace_name = 'AlexanderOpenAI64'
    # default_compute_target = "V100-32G"
    # ws = Workspace(subscription_id, resource_group, workspace_name)
    # ds = Datastore.get(ws, "babela100")
    # train_dataset = Dataset.File.from_files(path=[(ds, "github_data_fim/fim_megatron_github_dataset_300B/")], validate=True).as_mount()
    # train_dataset = Dataset.File.from_files(path=[(ds, "github_data_fim/fim_megatron_github_dataset_all/")], validate=True).as_mount()
    # vocab_dataset = Dataset.File.from_files(path=[( Datastore.get(ws, "workspaceblobstore"), "UI/2023-01-10_180934_UTC/")], validate=True).as_mount()

    # forced_interactive_auth = InteractiveLoginAuthentication(tenant_id="72f988bfc-86f1-41af-91ab-2d7cd011db47", force=True)
    # ws = Workspace(subscription_id, resource_group, workspace_name, auth=forced_interactive_auth)
    

    
    # subscription_id = 'a3e0a72d-17d7-45c6-a430-c8fdb8f3748d'
    # resource_group = 'DefaultResourceGroup-USW3'
    # workspace_name = 'CopilotGPUEastUS'
    # default_compute_target = "A10080G"
    # ws = Workspace(subscription_id, resource_group, workspace_name)
    # train_dataset = Dataset.get_by_name(ws, "megatron_github_300B")
    # vocab_dataset = Dataset.get_by_name(ws, "megatron-lm-vocab")
    

    
    subscription_id = '9ec1d932-0f3f-486c-acc6-e7d78b358f9b'
    resource_group = 'BabelReference'
    workspace_name = 'BabelEUSReference'
    default_compute_target = "A10080G"
    # ws = Workspace(subscription_id, resource_group, workspace_name, auth=forced_interactive_auth)
    ws = Workspace(subscription_id, resource_group, workspace_name)
    ds = Datastore.get(ws, "babela100")
    # train_dataset = Dataset.File.from_files(path=[(ds, "github_data_fim/fim_megatron_github_dataset_300B/")], validate=True).as_mount()
    train_dataset = Dataset.File.from_files(path=[(ds, "github_data_fim/fim_megatron_github_dataset_all/")], validate=True).as_mount()
    vocab_dataset = Dataset.File.from_files(path=[( Datastore.get(ws, "workspaceblobstore"), "UI/2023-01-22_054438_UTC/")], validate=True).as_mount()

    

    train_func = Component.from_yaml(
        ws,
        # yaml_file= Path(__file__).parent / "pretrain_gpt_moe.yaml"
        yaml_file= Path(__file__).parent / "train_moe.yaml"
        
    )
    
    

    
    # NUM_LAYERS=2
    # HIDDEN_SIZE=512
    # NUM_ATTN_HEADS=4
    # TRAIN_TOKENS=10
    # LR_DECAY_TOKENS=10
    # WARMUP_TOKENS=10
    # EVAL_INTERVAL=50
    # SAVE_INTERVAL=2500000
    # instance_count = 1


    
    NUM_LAYERS=24
    HIDDEN_SIZE=2048
    NUM_ATTN_HEADS=16
    
    # TRAIN_TOKENS=   300000000000
    TRAIN_TOKENS=   int(300000000000 / 100)
    LR_DECAY_TOKENS=TRAIN_TOKENS
    WARMUP_TOKENS=375000000
    EVAL_INTERVAL=1000
    SAVE_INTERVAL=20000
    # GLOBAL_BATCH_SIZE = 32

    
    # LOAD_BASE_PATH = Dataset.get_by_name(ws, "dense_checkpoint_test")
    # LOAD_BASE_PATH = Dataset.get_by_name(ws, "dense_checkpoint")

    LOAD_BASE_PATH = Dataset.File.from_files(path=[(ds, "github_data_fim/checkpoints_1.3b_rate_0.5_multihead_py/")], validate=True).as_mount()

    # EP_SIZE = 32
    EP_SIZE = 8

    instance_count = 1


    load_base_version = "v1"
    
    GLOBAL_BATCH_SIZE = int(4 * instance_count* 8)


    # timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')
    # output_path = f'github_moe_pretrain_misantac/logs-{timestamp}'
    @dsl.pipeline(
        name=f"megatron load_base={load_base_version} EP_SIZE={EP_SIZE} ",
        default_compute_target=default_compute_target,
        # default_compute_target="d15",

        default_datastore='workspaceblobstore',
    )
    def train_pipeline():
        trainer = train_func(
            # train_dataset = Dataset.get_by_name(ws, "megatron_github_300B"),
            # train_dataset = Dataset.File.from_files(path=[(ds, "github_data_fim/fim_megatron_github_dataset_all/")], validate=True).as_mount(),
            # train_dataset = Dataset.get_by_name(ws, "megatron_github_test"),

            train_dataset=train_dataset,
           

            # artifact_dir = Dataset.File.from_files(path=[(ds, "github_data_fim/artifacts/")], validate=True).as_mount(),
            # artifact_dir = Dataset.File.from_files(path=[(ds, "github_data_fim/artifacts_test/")], validate=True).as_mount(),

            # vocab_dataset = Dataset.get_by_name(ws, "megatron-lm-vocab"),
            # vocab_dataset = Dataset.File.from_files(path=[( Datastore.get(ws, "workspaceblobstore"), "UI/2023-01-10_180934_UTC/")], validate=True).as_mount(),

            vocab_dataset=vocab_dataset,

            


            # LOAD_PATH = Dataset.get_by_name(ws, "dummy_megatron_checkpoint"),

            # LOAD_BASE_PATH = Dataset.get_by_name(ws, "dummy_megatron_checkpoint"),
            # LOAD_BASE_PATH = Dataset.get_by_name(ws, "dummy_megatron_checkpoint_v2"),

            LOAD_BASE_PATH=LOAD_BASE_PATH,

            load_base_version=load_base_version,

            EP_SIZE=EP_SIZE,


            NUM_LAYERS=NUM_LAYERS,
            HIDDEN_SIZE=HIDDEN_SIZE,
            NUM_ATTN_HEADS=NUM_ATTN_HEADS,

            NUM_GPUS=8,

            TRAIN_TOKENS=TRAIN_TOKENS,
            WARMUP_TOKENS=WARMUP_TOKENS,
            LR_DECAY_TOKENS=LR_DECAY_TOKENS,
            EVAL_INTERVAL=EVAL_INTERVAL,
            SAVE_INTERVAL=SAVE_INTERVAL,


            
            GLOBAL_BATCH_SIZE=GLOBAL_BATCH_SIZE,

        )

        trainer.runsettings.resource_layout.configure(instance_count=instance_count, process_count_per_node=8)


        # trainer.outputs.save_path.configure(
        #     mode="upload",
        #     path_on_datastore=output_path,
        #     datastore=ds
        # )
        
        # trainer.outputs.artifact_dir.configure(
        #     mode="upload",
        #     path_on_datastore="github_data_fim/artifacts_test/",
        #     datastore=ds
        # )

    pipeline = train_pipeline()
    _ = pipeline.submit(
        experiment_name="misantac_megatron_pretrain",
        continue_on_step_failure=False,
    )


if __name__ == "__main__":
    main()
