#!/bin/bash

##########################################################

# The dir of genetated videos
VIDEO_DIR="/home/Mount_2_6T/dj_sora_challenge/output/generated_videos"

# Output dir of eval results
EVAL_OUTPUT_DIR="./evaluation_results_second_round_final_41k_full_512_ga2"

##########################################################

if [ -e "$EVAL_OUTPUT_DIR" ]; then
    echo "WARNING: Please make sure that your output dir contains only one evaluataion!"
fi

# The path to the models used by vbench
export VBENCH_CACHE_DIR=vbench_models/vbench

# prompts
PROMPT_INFO_JSON="VBench_sub_info.json"
#PROMPT_INFO_JSON="/home/Mount_2_6T/dj_sora_challenge/toolkit/evaluation/tion_test_output_json/motion.json"

# Define the dimension list
DIMENSIONS=("subject_consistency" "background_consistency" "dynamic_degree" "aesthetic_quality" "multiple_objects" "human_action" "scene" "overall_consistency" "imaging_quality")
#DIMENSIONS=("dynamic_degree")

TOTAL=${#DIMENSIONS[@]}

# Loop over each dimension
for i in "${!DIMENSIONS[@]}"; do
    # Get the dimension and corresponding folder
    DIMENSION=${DIMENSIONS[i]}
    # Calculate the progress
    PROGRESS=$((i+1))
    echo "$DIMENSION"
    python evaluate.py --videos_path $VIDEO_DIR --dimension $DIMENSION --output_path $EVAL_OUTPUT_DIR --full_json_dir "VBench_sub_info.json" --load_ckpt_from_local True
done

# copy for submission
rm -rf "../../../output/eval_results"
cp -r "$EVAL_OUTPUT_DIR" "../../../output/eval_results"

