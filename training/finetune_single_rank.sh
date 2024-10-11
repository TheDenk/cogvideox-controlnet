#!/bin/bash

export MODEL_PATH="THUDM/CogVideoX-2b"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# if you are not using wth 8 gus, change `accelerate_config_machine_single.yaml` num_processes as your gpu number
accelerate launch --config_file accelerate_config_machine_single.yaml --multi_gpu \
  train_controlnet.py \
  --tracker_name "cogvideox-controlnet" \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --validation_prompt "car is going in the ocean, beautiful waves:::ship in the vulcano" \
  --validation_video "../resources/car.mp4:::../resources/ship.mp4" \
  --validation_prompt_separator ::: \
  --num_inference_steps 28 \
  --num_validation_videos 1 \
  --validation_steps 500 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir "cogvideox-controlnet" \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --video_root_dir "set-path-to-video-directory" \
  --csv_path "set-path-to-csv-file" \
  --stride_min 1 \
  --stride_max 3 \
  --hflip_p 0.5 \
  --controlnet_type "canny" \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 3 \
  --downscale_coef 8 \
  --controlnet_weights 0.5 \
  --init_from_transformer \
  --train_batch_size 1 \
  --dataloader_num_workers 0 \
  --num_train_epochs 1 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-5 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 
  # --report_to wandb
  # --pretrained_controlnet_path "cogvideox-controlnet-2b/checkpoint-2000.pt" \
    