# Improvements to Dreamfusion's text-guided 3D shape generation model
##Promptist: reinforcement learning for automatic prompt optimization
LMOps is a research initiative on fundamental research and technology for building AI products w/ foundation models, especially on the general technology for enabling AI capabilities w/ LLMs and Generative AI models.
![image](https://user-images.githubusercontent.com/1070872/207856962-02f08d92-f2bf-441a-b1c3-efff1a4b6187.png)
###Load Pretrained Model for Stable Diffusion v1.4
```
import gradio as grad
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_prompter():
  prompter_model = AutoModelForCausalLM.from_pretrained("microsoft/Promptist")
  tokenizer = AutoTokenizer.from_pretrained("gpt2")
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "left"
  return prompter_model, tokenizer

prompter_model, prompter_tokenizer = load_prompter()

def generate(plain_text):
    input_ids = prompter_tokenizer(plain_text.strip()+" Rephrase:", return_tensors="pt").input_ids
    eos_id = prompter_tokenizer.eos_token_id
    outputs = prompter_model.generate(input_ids, do_sample=False, max_new_tokens=75, num_beams=8, num_return_sequences=8, eos_token_id=eos_id, pad_token_id=eos_id, length_penalty=-1.0)
    output_texts = prompter_tokenizer.batch_decode(outputs, skip_special_tokens=True)
    res = output_texts[0].replace(plain_text+" Rephrase:", "").strip()
    return res

txt = grad.Textbox(lines=1, label="Initial Text", placeholder="Input Prompt")
out = grad.Textbox(lines=1, label="Optimized Prompt")
examples = ["A rabbit is wearing a space suit", "Several railroad tracks with one train passing by", "The roof is wet from the rain", "Cats dancing in a space club"]

grad.Interface(fn=generate,
               inputs=txt,
               outputs=out,
               title="Promptist Demo",
               description="Promptist is a prompt interface for Stable Diffusion v1-4 (https://huggingface.co/CompVis/stable-diffusion-v1-4) that optimizes user input into model-preferred prompts.",
               examples=examples,
               allow_flagging='never',
               cache_examples=False,
               theme="default").launch(enable_queue=True, debug=True)
```
###Environment Setup
```
alias=`whoami | cut -d'.' -f2`; docker run -it --rm --runtime=nvidia --ipc=host --privileged -v /home/${alias}:/home/${alias} chizewen/pytorch:1.12.1-mpi bash
```
```
pip install git+https://github.com/CZWin32768/accelerate.git
pip install pytorch_lightning==1.7.7
pip install transformers==4.23.1
pip install ftfy regex tqdm scipy
pip install git+https://github.com/openai/CLIP.git
pip install --editable ./diffusers
cd trlx
pip install --editable .
cd ..
# please provide the access token of huggingface and your wandb key
```
###Train Promptist
```
python ./diffusers_examples/quick-start.py

accelerate launch --multi_gpu --machine_rank ${OMPI_COMM_WORLD_RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} --num_machines 4 --num_processes 32 ./diff_prompter/ppo_prompter.py --data /data_path --gpt_path /supervised_finetuned_gpt_path --trl_config ./diff_prompter/configs/ppo_config_a100_coco_bsz256_kl0.2.yml --checkpoint_dir /ckpt_dir
```


##Stable-Dreamfusion
A pytorch implementation of the text-to-3D model Dreamfusion, powered by the Stable Diffusion text-to-2D model.

https://user-images.githubusercontent.com/25863658/236712982-9f93bd32-83bf-423a-bb7c-f73df7ece2e3.mp4

https://user-images.githubusercontent.com/25863658/232403162-51b69000-a242-4b8c-9cd9-4242b09863fa.mp4
### Installation
```
git clone https://github.com/ashawkey/stable-dreamfusion.git
cd stable-dreamfusion
```
#### Optional: create a python virtual environment
To avoid python package conflicts, we recommend using a virtual environment, e.g.: using conda or venv:

```bash
python -m venv venv_stable-dreamfusion
source venv_stable-dreamfusion/bin/activate # you need to repeat this step for every new terminal
```
#### Install with pip
```bash
pip install -r requirements.txt
```
#### Download pre-trained models

To use image-conditioned 3D generation, you need to download some pretrained checkpoints manually:
* [Zero-1-to-3](https://github.com/cvlab-columbia/zero123) for diffusion backend.
    We use `105000.ckpt` by default, and it is hard-coded in `guidance/zero123_utils.py`.
    ```bash
    cd pretrained/zero123
    wget https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt
    ```
* [Omnidata](https://github.com/EPFL-VILAB/omnidata/tree/main/omnidata_tools/torch) for depth and normal prediction.
    These ckpts are hardcoded in `preprocess_image.py`.
    ```bash
    mkdir pretrained/omnidata
    cd pretrained/omnidata
    # assume gdown is installed
    gdown '1Jrh-bRnJEjyMCS7f-WsaFlccfPjJPPHI&confirm=t' # omnidata_dpt_depth_v2.ckpt
    gdown '1wNxVO4vVbDEMEpnAi_jwQObf2MFodcBR&confirm=t' # omnidata_dpt_normal_v2.ckpt
    ```

To use [DeepFloyd-IF](https://github.com/deep-floyd/IF), you need to accept the usage conditions from [hugging face](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), and login with `huggingface-cli login` in command line.

For DMTet, we port the pre-generated `32/64/128` resolution tetrahedron grids under `tets`.
The 256 resolution one can be found [here](https://drive.google.com/file/d/1lgvEKNdsbW5RS4gVxJbgBS4Ac92moGSa/view?usp=sharing).

#### Build extension (optional)
By default, we use [`load`](https://pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.load) to build the extension at runtime.
We also provide the `setup.py` to build each extension:
```bash
cd stable-dreamfusion

# install all extension modules
bash scripts/install_ext.sh

# if you want to install manually, here is an example:
pip install ./raymarching # install to python path (you still need the raymarching/ folder, since this only installs the built extension.)
```

#### Taichi backend (optional)
Use [Taichi](https://github.com/taichi-dev/taichi) backend for Instant-NGP. It achieves comparable performance to CUDA implementation while **No CUDA** build is required. Install Taichi with pip:
```bash
pip install -i https://pypi.taichi.graphics/simple/ taichi-nightly
```

#### Trouble Shooting:
* we assume working with the latest version of all dependencies, if you meet any problems from a specific dependency, please try to upgrade it first (e.g., `pip install -U diffusers`). If the problem still holds, [reporting a bug issue](https://github.com/ashawkey/stable-dreamfusion/issues/new?assignees=&labels=bug&template=bug_report.yaml&title=%3Ctitle%3E) will be appreciated!
* `[F glutil.cpp:338] eglInitialize() failed Aborted (core dumped)`: this usually indicates problems in OpenGL installation. Try to re-install Nvidia driver, or use nvidia-docker as suggested in https://github.com/ashawkey/stable-dreamfusion/issues/131 if you are using a headless server.
* `TypeError: xxx_forward(): incompatible function arguments`： this happens when we update the CUDA source and you used `setup.py` to install the extensions earlier. Try to re-install the corresponding extension (e.g., `pip install ./gridencoder`).

#### Tested environments
* Ubuntu 22 with torch 1.12 & CUDA 11.6 on a V100.

### Usage

First time running will take some time to compile the CUDA extensions.

```bash
#### stable-dreamfusion setting

### Instant-NGP NeRF Backbone
# + faster rendering speed
# + less GPU memory (~16G)
# - need to build CUDA extensions (a CUDA-free Taichi backend is available)

## train with text prompt (with the default settings)
# `-O` equals `--cuda_ray --fp16`
# `--cuda_ray` enables instant-ngp-like occupancy grid based acceleration.
python main.py --text "a hamburger" --workspace trial -O

# reduce stable-diffusion memory usage with `--vram_O`
# enable various vram savings (https://huggingface.co/docs/diffusers/optimization/fp16).
python main.py --text "a hamburger" --workspace trial -O --vram_O

# You can collect arguments in a file. You can override arguments by specifying them after `--file`. Note that quoted strings can't be loaded from .args files...
python main.py --file scripts/res64.args --workspace trial_awesome_hamburger --text "a photo of an awesome hamburger"

# use CUDA-free Taichi backend with `--backbone grid_taichi`
python3 main.py --text "a hamburger" --workspace trial -O --backbone grid_taichi

# choose stable-diffusion version (support 1.5, 2.0 and 2.1, default is 2.1 now)
python main.py --text "a hamburger" --workspace trial -O --sd_version 1.5

# use a custom stable-diffusion checkpoint from hugging face:
python main.py --text "a hamburger" --workspace trial -O --hf_key andite/anything-v4.0

# use DeepFloyd-IF for guidance (experimental):
python main.py --text "a hamburger" --workspace trial -O --IF
python main.py --text "a hamburger" --workspace trial -O --IF --vram_O # requires ~24G GPU memory

# we also support negative text prompt now:
python main.py --text "a rose" --negative "red" --workspace trial -O

## after the training is finished:
# test (exporting 360 degree video)
python main.py --workspace trial -O --test
# also save a mesh (with obj, mtl, and png texture)
python main.py --workspace trial -O --test --save_mesh
# test with a GUI (free view control!)
python main.py --workspace trial -O --test --gui

### Vanilla NeRF backbone
# + pure pytorch, no need to build extensions!
# - slow rendering speed
# - more GPU memory

## train
# `-O2` equals `--backbone vanilla`
python main.py --text "a hotdog" --workspace trial2 -O2

# if CUDA OOM, try to reduce NeRF sampling steps (--num_steps and --upsample_steps)
python main.py --text "a hotdog" --workspace trial2 -O2 --num_steps 64 --upsample_steps 0

## test
python main.py --workspace trial2 -O2 --test
python main.py --workspace trial2 -O2 --test --save_mesh
python main.py --workspace trial2 -O2 --test --gui # not recommended, FPS will be low.

### DMTet finetuning

## use --dmtet and --init_with <nerf checkpoint> to finetune the mesh at higher reslution
python main.py -O --text "a hamburger" --workspace trial_dmtet --dmtet --iters 5000 --init_with trial/checkpoints/df.pth

## init dmtet with a mesh to generate texture
# require install of cubvh: pip install git+https://github.com/ashawkey/cubvh
# remove --lock_geo to also finetune geometry, but performance may be bad.
python main.py -O --text "a white bunny with red eyes" --workspace trial_dmtet_mesh --dmtet --iters 5000 --init_with ./data/bunny.obj --lock_geo

## test & export the mesh
python main.py -O --text "a hamburger" --workspace trial_dmtet --dmtet --iters 5000 --test --save_mesh

## gui to visualize dmtet
python main.py -O --text "a hamburger" --workspace trial_dmtet --dmtet --iters 5000 --test --gui

### Image-conditioned 3D Generation

## preprocess input image
# note: the results of image-to-3D is dependent on zero-1-to-3's capability. For best performance, the input image should contain a single front-facing object, it should have square aspect ratio, with <1024 pixel resolution. Check the examples under ./data.
# this will exports `<image>_rgba.png`, `<image>_depth.png`, and `<image>_normal.png` to the directory containing the input image.
python preprocess_image.py <image>.png
python preprocess_image.py <image>.png --border_ratio 0.4 # increase border_ratio if the center object appears too large and results are unsatisfying.

## zero123 train
# pass in the processed <image>_rgba.png by --image and do NOT pass in --text to enable zero-1-to-3 backend.
python main.py -O --image <image>_rgba.png --workspace trial_image --iters 5000

# if the image is not exactly front-view (elevation = 0), adjust default_polar (we use polar from 0 to 180 to represent elevation from 90 to -90)
python main.py -O --image <image>_rgba.png --workspace trial_image --iters 5000 --default_polar 80

# by default we leverage monocular depth estimation to aid image-to-3d, but if you find the depth estimation inaccurate and harms results, turn it off by:
python main.py -O --image <image>_rgba.png --workspace trial_image --iters 5000 --lambda_depth 0

python main.py -O --image <image>_rgba.png --workspace trial_image_dmtet --dmtet --init_with trial_image/checkpoints/df.pth

## zero123 with multiple images
python main.py -O --image_config config/<config>.csv --workspace trial_image --iters 5000

## render <num> images per batch (default 1)
python main.py -O --image_config config/<config>.csv --workspace trial_image --iters 5000 --batch_size 4

# providing both --text and --image enables stable-diffusion backend (similar to make-it-3d)
python main.py -O --image hamburger_rgba.png --text "a DSLR photo of a delicious hamburger" --workspace trial_image_text --iters 5000

python main.py -O --image hamburger_rgba.png --text "a DSLR photo of a delicious hamburger" --workspace trial_image_text_dmtet --dmtet --init_with trial_image_text/checkpoints/df.pth

## test / visualize
python main.py -O --image <image>_rgba.png --workspace trial_image_dmtet --dmtet --test --save_mesh
python main.py -O --image <image>_rgba.png --workspace trial_image_dmtet --dmtet --test --gui

### Debugging

# Can save guidance images for debugging purposes. These get saved in trial_hamburger/guidance.
# Warning: this slows down training considerably and consumes lots of disk space!
python main.py --text "a hamburger" --workspace trial_hamburger -O --vram_O --save_guidance --save_guidance_interval 5 # save every 5 steps
```
For example commands, check [`scripts`](./scripts).

For advanced tips and other developing stuff, check [Advanced Tips](./assets/advanced.md).

### Evalutation

Reproduce the paper CLIP R-precision evaluation

After the testing part in the usage, the validation set containing projection from different angle is generated. Test the R-precision between prompt and the image.(R=1)

```bash
python r_precision.py --text "a snake is flying in the sky" --workspace snake_HQ --latest ep0100 --mode depth --clip clip-ViT-B-16
```
