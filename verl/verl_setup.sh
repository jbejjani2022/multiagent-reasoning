"Install from custom environment"
- https://verl.readthedocs.io/en/latest/start/install.html

loadpy
mamba create --name verl python==3.10 pip
start verl


# module spider - then find available cuda and gcc versions:
# cuda: cuda/9.1.85-fasrc01, cuda/11.3.1-fasrc01, cuda/11.8.0-fasrc01, cuda/12.0.1-fasrc01, cuda/12.2.0-fasrc01, cuda/12.4.1-fasrc01
#    Module for CUDA libraries
# gcc: gcc/9.5.0-fasrc01, gcc/10.2.0-fasrc01, gcc/12.2.0-fasrc01, gcc/13.2.0-fasrc01, gcc/14.2.0-fasrc01
#    the GNU Compiler Collection version 9.3.0
# cudnn: cudnn/8.8.0.121_cuda12-fasrc01, cudnn/8.9.2.26_cuda11-fasrc01, cudnn/8.9.2.26_cuda12-fasrc01, cudnn/9.1.1.17_cuda12-fasrc01, ...
#    The NVIDIA CUDA Deep Neural Network library (cuDNN) is a GPU-accelerated library of primitives for deep neural networks.


# module load cuda/12.4.1-fasrc01
# module load cuda/12.2.0-fasrc01
module load cuda
module load cudnn
module load gcc/12.2.0-fasrc01
# module load cudnn/9.1.1.17_cuda12-fasrc01

pip install torch==2.6.0+cu124 torchvision==0.19.1+cu124 --index-url https://download.pytorch.org/whl/cu124

pip install torch torchvision
pip install flash-attn --no-build-isolation
git clone https://github.com/volcengine/verl.git
cd verl
