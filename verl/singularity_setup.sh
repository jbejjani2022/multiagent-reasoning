# https://docs.sylabs.io/guides/2.6/user-guide/singularity_and_docker.html
# https://pmitev.github.io/UPPMAX-Singularity-workshop/docker2singularity/

# start a light interactive session
salloc --partition=test --account=kempner_sham_lab --nodes=1 --ntasks-per-node=4 --mem-per-cpu=3200M --time=02:00:00

# for vLLM with Megatron or FSDP
# docker image is at: whatcanyousee/verl:ngc-cu124-vllm0.8.3-sglang0.4.5-mcore0.12.0-te2.2
# hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0

cd verl
# didn't work: singularity pull docker://whatcanyousee/verl:ngc-cu124-vllm0.8.3-sglang0.4.5-mcore0.12.0-te2.2
# did instead:
singularity pull docker://hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0

# open an interactive shell with the image active
singularity shell --nv verl_ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0.sif
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
exit

# Inside the container, install latest verl:
# install the nightly version (recommended)
git clone https://github.com/volcengine/verl && cd verl
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
pip3 install --no-build-isolation -e .[vllm]  # or -e .[sglang]


# run the singularity image
# --nv makes the NVIDIA GPU and its drivers from the host machine available inside the container
singularity run --nv verl_ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0.sif


# execute a script
singularity exec --nv verl_ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0.sif python script.py

# Singularity> python -c "import torch, flash_attn; print(f'torch version: {torch.__version__}\nflash-attn version: {flash_attn.__version__}')"
# torch version: 2.6.0+cu124
# flash-attn version: 2.7.4.post1
