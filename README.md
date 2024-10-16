# CogvideoX Controlnet Extention


https://github.com/user-attachments/assets/34a3a015-bfd3-4f75-a6ba-e8be524916e2

https://github.com/user-attachments/assets/596ca9ce-198c-476c-8ace-077c2a534ae4

This repo contains the code for simple Controlnet module for CogvideoX model.  
Supported models for 2B:
- Canny (<a href="https://huggingface.co/TheDenk/cogvideox-2b-controlnet-canny-v1">HF Model Link</a>) 
- Hed (<a href="https://huggingface.co/TheDenk/cogvideox-2b-controlnet-hed-v1">HF Model Link</a>) 

### How to
Clone repo 
```bash
git clone https://github.com/TheDenk/cogvideox-controlnet.git
cd cogvideox-controlnet
```
  
Create venv  
```bash
python -m venv venv
source venv/bin/activate
```
  
Install requirements
```bash
pip install -r requirements.txt
```
  
### Simple examples
#### Inference with cli
```bash
python -m inference.cli_demo \
    --video_path "resources/car.mp4" \
    --prompt "car is moving among mountains" \
    --controlnet_type "canny" \
    --base_model_path THUDM/CogVideoX-2b \
    --controlnet_model_path TheDenk/cogvideox-2b-controlnet-canny-v1
```

#### Inference with Gradio
```bash
python -m inference.gradio_web_demo \
    --controlnet_type "canny" \
    --base_model_path THUDM/CogVideoX-2b \
    --controlnet_model_path TheDenk/cogvideox-2b-controlnet-canny-v1
```

### Detailed inference
```bash
python -m inference.cli_demo \
    --video_path "resources/car.mp4" \
    --prompt "car is moving on waves in the ocean" \
    --controlnet_type "canny" \
    --base_model_path THUDM/CogVideoX-2b \
    --controlnet_model_path TheDenk/cogvideox-2b-controlnet-canny-v1 \
    --num_inference_steps 50 \
    --guidance_scale 6.0 \
    --controlnet_weights 0.8 \
    --controlnet_guidance_start 0.0 \
    --controlnet_guidance_end 0.8 \
    --output_path "./output.mp4" \
    --seed 42
```

## Training
The 2B model requires 48 GB VRAM (For example A6000) and 80 GB for 5B. But it depends on the number of transformer blocks which default is 8 (`controlnet_transformer_num_layers` parameter in the config).

#### Dataset
<a href="https://huggingface.co/datasets/nkp37/OpenVid-1M">OpenVid-1M</a> dataset was taken as the base variant. CSV files for the dataset you can find <a href="https://huggingface.co/datasets/nkp37/OpenVid-1M/tree/main/data/train">here</a>.

#### Train script
For start training you need fill the config files `accelerate_config_machine_single.yaml` and `finetune_single_rank.sh`.  
In `accelerate_config_machine_single.yaml` set parameter`num_processes: 1` to your GPU count.  
In `finetune_single_rank.sh`:  
1. Set `MODEL_PATH for` base CogVideoX model. Default is THUDM/CogVideoX-2b.  
2. Set `CUDA_VISIBLE_DEVICES` (Default is 0).  
3. (For OpenVid dataset) Set `video_root_dir` to directory with video files and `csv_path`.  

Run taining
```
cd training
bash finetune_single_rank.sh
```

## Acknowledgements
Original code and models [CogVideoX](https://github.com/THUDM/CogVideo/tree/main).  

## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>
