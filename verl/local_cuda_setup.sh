mkdir -p $HOME/.local/cuda-12.4
cd $HOME/.local/cuda-12.4

# wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.14_linux.run
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.40.07_linux.run


sh cuda_12.4.1_550.54.14_linux.run --silent --toolkit --installpath=$HOME/.local/cuda-12.4

export CUDA_HOME=$HOME/.local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ls $CUDA_HOME/lib64/libcusparseLt.so*

pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
