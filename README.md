# Generative-Spatial-Control
Research project for the seminar "Generative Visual Models".




## Setup instructions

### Clone repository:

`git clone --recurse-submodules git@github.com:TheBlueHawk/Generative-Spatial-Control.git`

### Euler install
Login to Euler:
```bash
ssh {eth-username}@euler.ethz.ch
```
Setup ssh keys with github and add to ssh-agent:
```bash
eval `ssh-agent`
ssh-add {path-to-github-ssh-key}
```
Setup euler:
```bash
python3 -m venv venv
source venv/bin/activate
env2lmod
module load gcc/8.2.0 python_gpu/3.10.4 r/4.0.2 git-lfs/2.3.0 eth_proxy npm/6.14.9
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
pip install -U xformers
export PYTHONPATH="${PYTHONPATH}:/cluster/home/{ethz-username}/Generative-Spatial-Control"
```

### Submit jobs
#### Interactive shell
```bash
srun --gpus=1 --ntasks=4 --mem-per-cpu=4G --pty bash
```


#### Batch
([gpu instruction](https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs), [euler batch](https://scicomp.ethz.ch/wiki/Using_the_batch_system)):
```bash
bsub -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -oo log.out -eo error.out -W 23:59 -B -N <your_python_command>
```
