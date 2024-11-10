CUDA_VISIBLE_DEVICES=0 python run_inference_benchmark_dramaqa.py \
    --model_path checkpoints/videollava-7b-lora-epoch-front/checkpoint-2260 \
    --cache_dir cache_dir \
    --video_dir data/AnotherMissOh_images/dramaqa_frames \
    --gt_file data/llava_dramaqg_val.json \
    --output_dir preds-e4-val \
    --output_name pred \
    --device cuda:0 \
    --model_base lmsys/vicuna-7b-v1.5 \