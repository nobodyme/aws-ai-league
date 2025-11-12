# !/bin/bash

# List all codes for SageMaker service quotas
# aws service-quotas list-service-quotas --service-code sagemaker | grep -C 10 ml.g5.48xlarge

# Request a quota increase for ml.g5.48xlarge instances for training jobs
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-6BC98A55 \
  --desired-value 4

# Request a quota increase for ml.g5.12xlarge instances for endpoint usage
aws service-quotas request-service-quota-increase \
  --service-code sagemaker \
  --quota-code L-65C4BD00 \
  --desired-value 4