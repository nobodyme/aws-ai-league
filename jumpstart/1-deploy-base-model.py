from sagemaker.jumpstart.model import JumpStartModel

# Looks like 70B model deployment rquires p4d instance which is not available in my account.
# ResourceLimitExceeded: An error occurred (ResourceLimitExceeded) when calling the CreateEndpoint operation: The 
# account-level service limit 'ml.p4d.24xlarge for endpoint usage' is 0 Instances, with current utilization of 0 
# Instances and a request delta of 1 Instances. Please use AWS Service Quotas to request an increase for this quota. 
# If AWS Service Quotas is not available, contact AWS support to request an increase for this quota.
# TODO: Move to deployed bedrock instance - do the necessary prompting changes for evaluation
# NOTE: Not sure deployed bedrock instance will be available in the given account

def deploy_model(model_id, model_version):
    pretrained_model = JumpStartModel(model_id=model_id, model_version=model_version)
    pretrained_predictor = pretrained_model.deploy(accept_eula=True)
    return pretrained_predictor

def cleanup_model(pretrained_predictor):
    pretrained_predictor.delete_model()
    pretrained_predictor.delete_endpoint()


if __name__ == "__main__":
    model_id, model_version = "meta-textgeneration-llama-3-8b", "2.*"
    pretrained_predictor = deploy_model(model_id, model_version)
    print(f"Deployed model endpoint: {pretrained_predictor.endpoint_name}")
    # Uncomment the following line to list all available models
    # list_models()
    # Uncomment the following line to clean up the deployed model and endpoint
    # cleanup_model(pretrained_predictor)