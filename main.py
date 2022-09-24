#import required libraries
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext
from azure.ai.ml.entities import AmlCompute
from azure.ai.ml import command, Input

#Enter details of your AzureML workspace
subscription_id = '7b7e94ec-f305-433f-829b-475500f5b6b3'
resource_group = 'KS_hack_2022'
workspace = 'KS_reactor_hack'

#connect to the workspace
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)

# specify aml compute name.
compute_instance = "rlsri23051"

env_docker_conda = Environment(
    image="mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04",
    conda_file="environment.yml",
    name="docker-image-plus-conda-example",
    description="Environment created from a Docker image plus Conda environment.",
)
ml_client.environments.create_or_update(env_docker_conda)

# define the command
command_job = command(
    code="./",
    command="python crop_predict.py --crop-csv ${{inputs.crop_csv}}",
    environment=env_docker_conda,
    inputs={
        "crop_csv": Input(
            type="uri_file",
            path="Crop_recommendation.csv",
        )
    },
    compute= compute_instance,
)



# submit the command
returned_job = ml_client.jobs.create_or_update(command_job)

print("success")

# get a URL for the status of the job
returned_job.services["Studio"].endpoint