"""Deploy a fine-tuned JumpStart model using only its training job name.

Populate the configuration block under the ``__main__`` guard with the
JumpStart training job name produced during fine-tuning. The script will:

1. Attach to the completed training job via ``JumpStartEstimator.attach``.
2. Create a SageMaker model from the fine-tuned artifacts.
3. Deploy it to an endpoint (default ``<training-job>-endpoint`` unless you
   override ``ENDPOINT_NAME``).

The resulting ``predictor`` is returned so you can invoke it or clean it up.
"""

from __future__ import annotations

from typing import Optional

from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.deserializers import JSONDeserializer
from sagemaker.serializers import JSONSerializer


def deploy_finetuned_model(training_job_name: str):
    """Attach to a completed JumpStart training job and deploy its model.

    Parameters
    ----------
    training_job_name:
        Name of the SageMaker training job returned by fine-tuning.
    """

    estimator = JumpStartEstimator.attach(training_job_name=training_job_name)

    resolved_endpoint_name = f"{training_job_name}-endpoint"
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type="ml.g5.12xlarge",
        endpoint_name=resolved_endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        wait=False
    )

    print(f"Deployed endpoint: {predictor.endpoint_name}")
    return predictor


if __name__ == "__main__":
    TRAINING_JOB_NAMES = ["jumpstart-ft-2-eff48b5d", "jumpstart-ft-1-43ae7938"] 

    if not TRAINING_JOB_NAMES:
        raise SystemExit("Set TRAINING_JOB_NAME to the fine-tuning job you want to deploy.")

    endpoints = []
    for job_name in TRAINING_JOB_NAMES:
        endpoints.append(deploy_finetuned_model(
            training_job_name=job_name
        ))

    print("Deployed endpoints:", [ep.endpoint_name for ep in endpoints])