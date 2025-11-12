## This can only be run from jupyter notebook inside sagemaker
from __future__ import annotations
from typing import Dict
from uuid import uuid4
from sagemaker.jumpstart.estimator import JumpStartEstimator
from utils.log import log_run


def launch_fine_tuning_job(
    model_id: str,
    model_version: str,
    train_data_location: str,
    hyperparameters: Dict[str, object],
    wait: bool = False,
    logs: bool = False,
    job_name: str | None = None,
):
    """Kick off a single JumpStart fine-tuning job and return its estimator."""

    estimator = JumpStartEstimator(
        model_id=model_id,
        model_version=model_version,
        environment={"accept_eula": "true"},
        disable_output_compression=True,
        instance_type="ml.g5.48xlarge",
    )

    # JumpStart expects hyperparameters as strings.
    stringified_hyperparameters = {key: str(value) for key, value in hyperparameters.items()}
    estimator.set_hyperparameters(**stringified_hyperparameters)

    estimator.fit(
        {"training": train_data_location},
        job_name=job_name,
        wait=wait,
        logs=logs,
    )

    return estimator


if __name__ == "__main__":
    model_id, model_version = "meta-textgeneration-llama-3-8b-instruct", "2.20.1"
    train_data_location = (
        "s3://sagemaker-us-east-1-466279506647/dolly_dataset/train.jsonl"
    )

    parameter_sets = [
        {
            "epoch": "1",
            "learning_rate": "1e-4",
            "lora_r": "4",
            "lora_alpha": "8",
            "target_modules": "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj",
            "lora_dropout": "0.05",
            "instruction_tuned": "True",
            "chat_dataset": "False",
            "per_device_train_batch_size": "4",
            # "per_device_eval_batch_size": "1",
            # "max_train_samples": "-1",
            # "max_val_samples": "-1",
            "seed": "42",
            "max_input_length": "1024",
            # "validation_split_ratio:": "0.2",
            # "train_data_split_seed": 0
            "preprocessing_num_workers": "2",
            "int8_quantization": "False",
            "enable_fsdp": "True",
        },
        {
            "epoch": "3",
            "learning_rate": "5e-5",
            "instruction_tuned": "True",
            "chat_dataset": "False",
            "max_input_length": "1024",
            "per_device_train_batch_size": "2",
        },
    ]

    launched_jobs = []
    job_name_prefix = "jumpstart-ft"

    for index, params in enumerate(parameter_sets, start=1):
        unique_suffix = uuid4().hex[:8]
        job_name = f"{job_name_prefix}-{index}-{unique_suffix}"[:63]

        print(f"Launching training job '{job_name}' with params: {params}")

        estimator = launch_fine_tuning_job(
            model_id=model_id,
            model_version=model_version,
            train_data_location=train_data_location,
            hyperparameters=params,
            wait=False,
            logs=False,
            job_name=job_name,
        )

        job_record = {
            "job_name": job_name,
            "params": params,
            "train_data_location": train_data_location,
        }
        log_run(job_record, log_name="fine_tune_runs")

        print(f"Started training job '{job_name}' with params: {params}")
        launched_jobs.append({**job_record, "estimator": estimator})

    print("Launched the following training jobs:")
    for job in launched_jobs:
        print(f"  - {job['job_name']} (params: {job['params']})")

    # Outputs can be found under Jobs/Training in the SageMaker Console.