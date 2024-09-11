##########################################################
# dataset path
DATASET_NAME="../output/processed_data"
DATASET_META_NAME="../output/processed_data/recaption_second_round_motion_sort_41k.jsonl"

# training config
MIXED_PRECISION="bf16"
BATCH_SIZE_PER_GPU=1
GRADIENT_ACCUMULATION_STEPS=1
NUM_TRAIN_EPOCHS=3
DATALOADER_NUM_WORKERS=0

# saving config
OUTPUT_DIR="output_dir_second_round_final_recaption_41k_512_epoch3_ga1" # 模型输出文件存放文件夹
CHECKPOINTING_STEPS=10555 #2345*6, 15625 stop
VALIDATION_STEPS=5000000
VALIDATION_PROMPTS="A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff\'s precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures."
##########################################################



accelerate launch --mixed_precision=$MIXED_PRECISION scripts/train_lora.py \
  --config_path "config/easyanimate_video_motion_module_v1.yaml" \
  --pretrained_model_name_or_path="pretrained_models/Diffusion_Transformer/PixArt-XL-2-512x512" \
  --transformer_path="pretrained_models/Motion_Module/easyanimate_mm_16x512x512_pretrain.safetensors" \
  --train_data_dir=$DATASET_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --sample_size=512 \
  --sample_n_frames=16 \
  --sample_stride=2 \
  --train_batch_size=$BATCH_SIZE_PER_GPU \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --num_train_epochs=$NUM_TRAIN_EPOCHS \
  --dataloader_num_workers=$DATALOADER_NUM_WORKERS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --validation_prompts="$VALIDATION_PROMPTS" \
  --output_dir=$OUTPUT_DIR \
  --validation_steps=$VALIDATION_STEPS \
  --learning_rate=2e-05 \
  --seed=42 \
  --enable_xformers_memory_efficient_attention \
  --gradient_checkpointing \
  --adam_weight_decay=3e-2 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  2>&1 | tee ../output/train.log


# copy for submission
#cp "$0" "../../output/train_lora.sh"

# bash ./infer_lora_512.sh
