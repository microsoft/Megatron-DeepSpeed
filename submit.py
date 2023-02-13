
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



def main():
    # subscription_id = '9ec1d932-0f3f-486c-acc6-e7d78b358f9b'
    # resource_group = 'CopilotEyesOff'
    # workspace_name = 'copilotEyesOffWestUS3'
    # default_compute_target = "A100-40G-non-ipp"
    # ws = Workspace(subscription_id, resource_group, workspace_name)
    # ds = Datastore.get(ws, "blob_data_babela100")
    # train_dataset = Dataset.get_by_name(ws, "megatron_github_test")
    # vocab_dataset = Dataset.get_by_name(ws, "megatron-lm-vocab")

    
    subscription_id = '79f57c16-00fe-48da-87d4-5192e86cd047'
    resource_group = 'alexanderopenai64'
    workspace_name = 'AlexanderOpenAI64'
    default_compute_target = "V100-32G"
    ws = Workspace(subscription_id, resource_group, workspace_name)
    ds = Datastore.get(ws, "babela100")
    train_dataset = Dataset.File.from_files(path=[(ds, "github_data_fim/fim_megatron_github_dataset_all/")], validate=True).as_mount()
    vocab_dataset = Dataset.File.from_files(path=[( Datastore.get(ws, "workspaceblobstore"), "UI/2023-01-10_180934_UTC/")], validate=True).as_mount()
    

    
    # subscription_id = 'a3e0a72d-17d7-45c6-a430-c8fdb8f3748d'
    # resource_group = 'DefaultResourceGroup-USW3'
    # workspace_name = 'CopilotGPUEastUS'
    # default_compute_target = "A10080G"
    # ws = Workspace(subscription_id, resource_group, workspace_name)
    # ds = Datastore.get(ws, "babela100wus39060115481")

    
    # subscription_id = '9ec1d932-0f3f-486c-acc6-e7d78b358f9b'
    # resource_group = 'BabelReference-USW3'
    # workspace_name = 'BabelEUSReference'
    # default_compute_target = "A10080G"
    # ws = Workspace(subscription_id, resource_group, workspace_name)
    # ds = Datastore.get(ws, "babela100")

    train_func = Component.from_yaml(
        ws,
        # yaml_file= Path(__file__).parent / "pretrain_gpt_moe.yaml"
        yaml_file= Path(__file__).parent / "train_moe.yaml"
        
    )
    
    
    # .as_mount()


    # timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H%M%S')
    # output_path = f'github_moe_pretrain_misantac/logs-{timestamp}'
    @dsl.pipeline(
        name=f"megatron pretrain",
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


            # load_path = Dataset.get_by_name(ws, "test_checkpoint_load"),

            NUM_LAYERS=2,
            HIDDEN_SIZE=512,
            NUM_ATTN_HEADS=4,

            NUM_GPUS=8,


        )
        # trainer.runsettings.resource_layout.configure(instance_count=2, process_count_per_node=8)
        trainer.runsettings.resource_layout.configure(instance_count=1, process_count_per_node=8)


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
