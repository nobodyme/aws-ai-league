# create a 3.11 env and register as a kernel, run one by one
conda create -y -n py311 python=3.11
conda activate py311
python -m pip install -U pip ipykernel sagemaker
python -m ipykernel install --user --name py311 --display-name "Python 3.11 (py311)"

# NOTE:
# After this go the jupyter notebook and select the kernel -> change kernal to "Python 3.11 (py311)" to use this environment

