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
    --controlnet_weights 0.5 \
    --controlnet_guidance_start 0.0 \
    --controlnet_guidance_end 0.5 \
    --output_path "./output.mp4" \
    --seed 42
```

  
## Acknowledgements
Original code and models [CogVideoX](https://github.com/THUDM/CogVideo/tree/main).  

## Contacts
<p>Issues should be raised directly in the repository. For professional support and recommendations please <a>welcomedenk@gmail.com</a>.</p>
