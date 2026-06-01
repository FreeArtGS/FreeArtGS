# Installation

## Basic

```bash
git clone https://github.com/FreeArtGS/FreeArtGS.git
cd FreeArtGS
git submodule update --init --recursive
conda create -n freeartgs python=3.10 -y
conda activate freeartgs
# install torch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
```

## Preprocess

```bash
# loading checkpoint
python pipelines/preload_bert.py
mkdir -p checkpoints
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth -O checkpoints/groundingdino_swinb_cogcoor.pth
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt -O checkpoints/sam2_hiera_large.pt

cd preprocessing/GroundingDINO && python setup.py build && python setup.py install && cd ../..
cd preprocessing/segment-anything-2 && pip install . --verbose --no-build-isolation && python setup.py build_ext --inplace && cd ../..
```
Also, dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth under checkpoints/dinov3
## Reconstruction

### Nerfstudio

```bash
cd reconstruction/tiny-cuda-nn/bindings/torch/ && python setup.py install && cd ../../../..
pip install reconstruction/nerfacc_nerfstudio --verbose --no-build-isolation
pip install -e reconstruction/nerfstudio --no-build-isolation
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```
