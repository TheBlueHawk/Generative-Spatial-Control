# Cross-Attention Masking for Generative Spatial Control
This repository contains our work for the research project part of the seminar "Generative Visual Models". We focused on improving spatial control within Stable Diffusion models, addressing the common issue of object misplacement and window artifacts. We introduce a method that leverages cross-attention control in the latent space, aiming for more spatially coherent image generation.

## Setup instructions

### Clone repository:

`git clone --recurse-submodules git@github.com:TheBlueHawk/Generative-Spatial-Control.git`

### Env setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -U xformers
```

## Usage 
```bash
python script.py
```

Or just modify the Jupyter notebook.
