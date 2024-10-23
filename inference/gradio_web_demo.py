import argparse
import os
import threading
import time

import gradio as gr
import torch
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.utils import export_to_video, load_video
from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    AutoencoderKLCogVideoX
)
from datetime import datetime, timedelta
import moviepy.editor as mp
from controlnet_aux import HEDdetector, CannyDetector

from controlnet_pipeline import ControlnetCogVideoXPipeline
from cogvideo_transformer import CustomCogVideoXTransformer3DModel
from cogvideo_controlnet import CogVideoXControlnet


os.makedirs("./output", exist_ok=True)
os.makedirs("./gradio_tmp", exist_ok=True)

controlnet_mapping = {
    'hed': HEDdetector,
    'canny': CannyDetector,
}


def init_controlnet_processor(controlnet_type):
    if controlnet_type in ['canny', 'lineart']:
        return controlnet_mapping[controlnet_type]()
    return controlnet_mapping[controlnet_type].from_pretrained('lllyasviel/Annotators').to(device='cuda')


def save_video(tensor):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"./output/{timestamp}.mp4"
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    export_to_video(tensor, video_path)
    return video_path


def convert_to_gif(video_path):
    clip = mp.VideoFileClip(video_path)
    clip = clip.set_fps(8)
    clip = clip.resize(height=240)
    gif_path = video_path.replace(".mp4", ".gif")
    clip.write_gif(gif_path, fps=8)
    return gif_path


def delete_old_files():
    while True:
        now = datetime.now()
        cutoff = now - timedelta(minutes=10)
        directories = ["./output", "./gradio_tmp"]

        for directory in directories:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    if file_mtime < cutoff:
                        os.remove(file_path)
        time.sleep(600)


threading.Thread(target=delete_old_files, daemon=True).start()

def main(args):
    # 0. Load selected controlnet processor
    controlnet_processor = init_controlnet_processor(args.controlnet_type)
    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    tokenizer = T5Tokenizer.from_pretrained(
        args.base_model_path, subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        args.base_model_path, subfolder="text_encoder"
    )
    transformer = CustomCogVideoXTransformer3DModel.from_pretrained(
        args.base_model_path, subfolder="transformer"
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        args.base_model_path, subfolder="vae"
    )
    scheduler = CogVideoXDDIMScheduler.from_pretrained(
        args.base_model_path, subfolder="scheduler"
    )
    controlnet = CogVideoXControlnet.from_pretrained(
        args.controlnet_model_path
    )

    pipe = ControlnetCogVideoXPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        transformer=transformer,
        vae=vae,
        controlnet=controlnet,
        scheduler=scheduler,
    )
    
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path, weight_name="pytorch_lora_weights.safetensors", adapter_name="test_1")
        pipe.fuse_lora(lora_scale=1 / args.lora_rank)

    # pipe.scheduler = CogVideoXDDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # pipe.to("cuda")
    pipe = pipe.to(dtype=dtype)
    pipe.enable_sequential_cpu_offload()

    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()


    def infer(prompt: str, controlnet_frames: list, num_inference_steps: int, guidance_scale: float, seed: int, progress=gr.Progress(track_tqdm=True)):
        torch.cuda.empty_cache()
        video = pipe(
            prompt=prompt,
            controlnet_frames=controlnet_frames,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=49,
            guidance_scale=guidance_scale,
            generator=torch.Generator().manual_seed(seed),
        ).frames[0]

        return video

    with gr.Blocks() as demo:
        gr.Markdown("""
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                CogVideoX Controlnet Gradio Simple Space🤗
                """)

        with gr.Row():
            with gr.Column():
                with gr.Column():
                    video_input = gr.Video(label="Video for controlnet processing", width=720, height=480)
                    with gr.Row():
                        download_video_button = gr.File(label="📥 Download Video", visible=False)
                        download_gif_button = gr.File(label="📥 Download GIF", visible=False)
                prompt = gr.Textbox(label="Prompt (Less than 200 Words)", placeholder="Enter your prompt here", lines=5)

                with gr.Column():
                    gr.Markdown(
                        "**Optional Parameters** (default values are recommended)<br>"
                        "Increasing the number of inference steps will produce more detailed videos, but it will slow down the process.<br>"
                        "50 steps are recommended for most cases.<br>"
                    )
                    with gr.Row():
                        num_inference_steps = gr.Number(label="Inference Steps", value=50)
                        guidance_scale = gr.Number(label="Guidance Scale", value=6.0)
                        seed = gr.Number(label="Seed", value=42)
                    generate_button = gr.Button("🎬 Generate Video")

            with gr.Column():
                video_output = gr.Video(label="CogVideoX Generate Video", width=720, height=480)
                with gr.Row():
                    download_video_button = gr.File(label="📥 Download Video", visible=False)
                    download_gif_button = gr.File(label="📥 Download GIF", visible=False)

        def generate(prompt, video_input, num_inference_steps, guidance_scale, seed, progress=gr.Progress(track_tqdm=True)):
            video = load_video(video_input)[:49]
            controlnet_frames = [controlnet_processor(x) for x in video]
            tensor = infer(prompt, controlnet_frames, num_inference_steps, guidance_scale, seed, progress=progress)
            video_path = save_video(tensor)
            video_update = gr.update(visible=True, value=video_path)
            gif_path = convert_to_gif(video_path)
            gif_update = gr.update(visible=True, value=gif_path)

            return video_path, video_update, gif_update

        generate_button.click(
            generate,
            inputs=[prompt, video_input, num_inference_steps, guidance_scale, seed],
            outputs=[video_output, download_video_button, download_gif_button],
        )
    demo.launch()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument(
        "--base_model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument(
        "--controlnet_model_path", type=str, default="TheDenk/cogvideox-5b-controlnet-hed-v1", help="The path of the controlnet pre-trained model to be used"
    )
    parser.add_argument("--controlnet_type", type=str, default='hed', help="Type of controlnet model (e.g. canny, hed)")
    parser.add_argument("--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--lora_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    args = parser.parse_args()
    main(args)