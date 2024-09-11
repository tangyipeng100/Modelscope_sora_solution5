#!/bin/bash

##########################################################

# the path to your lora model
LORA_PATH="../output/lora_model/checkpoint-31164.safetensors" # 可使用训练权重，或者本方案下载权重，模型下载链接见readme文档

# inferance config
GPU_NUM=2
BATCH_SIZE=8
MIXED_PRECISION="bf16"

# saving config
OUTPUT_VIDEO_DIR="video_output_dir_second_round_final_41k_512_ga2"

##########################################################

VIDEO_NUM_PER_PROMPT=3


# run
for (( i = 0; i < GPU_NUM; i++ )); do
{
    CUDA_VISIBLE_DEVICES=$i python scripts/infer_lora.py \
      --prompt_info_path="./evaluation/VBench_sub_info.json" \
      --config_path "config/easyanimate_video_motion_module_v1.yaml" \
      --pretrained_model_name_or_path="pretrained_models/Diffusion_Transformer/PixArt-XL-2-512x512" \
      --transformer_path="pretrained_models/Motion_Module/easyanimate_mm_16x512x512_pretrain.safetensors" \
      --lora_path=$LORA_PATH \
      --image_size=512 \
      --chunks_num=$GPU_NUM \
      --chunk_id=$i \
      --batch_size=$BATCH_SIZE \
      --video_num_per_prompt=$VIDEO_NUM_PER_PROMPT \
      --mixed_precision=$MIXED_PRECISION \
      --save_path=$OUTPUT_VIDEO_DIR \
      --seed=43
} &   
done

wait


# copy for submission
cp "$0" "../output/infer_lora.sh"
cp "$LORA_PATH" "../output/lora_model"
rm -rf "../output/generated_videos"
cp -r "$OUTPUT_VIDEO_DIR" "../output/generated_videos"