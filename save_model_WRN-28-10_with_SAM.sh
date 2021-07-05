#!/bin/bash
#YBATCH -r any_1
#SBATCH -N 1
#SBATCH -J sam_test
#SBATCH --output output/%j.out

# ======== Pyenv/ ========
# export PYENV_ROOT=$HOME/.pyenv
# export PATH=$PYENV_ROOT/bin:$PATH
# eval "$(pyenv init -)"

# ======== Modules ========
. /etc/profile.d/modules.sh
module load cuda/11.1 cudnn/cuda-11.1/8.0 nccl/cuda-11.1/2.7.8 openmpi/3.1.6

export MASTER_ADDR=$(/usr/sbin/ip a show | grep inet | grep 192.168.205 | head -1 | cut -d " " -f 6 | cut -d "/" -f 1)
export MASTER_PORT=3535
# export LD_LIBRARY_PATH=~/.pyenv/versions/timm_3.8.6/lib/python3.8/site-packages/torch/lib/:$LD_LIBRARY_PATH
# For example: ~/.pyenv/versions/hoge/lib/python3.8/site-packages/torch/lib/:$LD_LIBRARY_PATH

# batch size := global mini-batch size (!= local mini-batch size)
# for example, when --batch-size 256 and NGPUS=8, local mini-batch size is 256/8 = 32

export NGPUS=1
export NUM_PROC=4

mpirun -npernode $NUM_PROC -np $NGPUS \
python save_model_WRN-28-10_with_SAM.py \
  --dist-url $MASTER_ADDR \
  --epochs 200 \
  --batch-size 4096 \
  --nbs 0.08 \
  --experiment bs_4096_ngpus_1_lr_0.1_nbs_0.08
