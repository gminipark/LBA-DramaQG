# LBA-Text2Data

## Testing
### Inference (BackBone: HF Transformers; LLaMA3.1-8B)
```
$ nohup python -u -m Test.eval_week8.py > ./Logs/TestOutput\(Week8\).log 2>&1 &

$ nohup python -u -m Test.eval_week9.py > ./Logs/TestOutput\(Week9\).log 2>&1 &
```

## Training
### Data-Building
```
$ python -m LLaMA3_1.fine_tuning.train_data_build
```

### Fine-Tuning
```bash
$ export RunID=LBA_LoRA_rank32_alpha16_epochs5_8B_ZeroShot

$ nohup python -u -m LLaMA3_1.fine_tuning.lora_train_llama \
    --dataset_text_field Text_ZeroShot \
    --base_model_path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --new_model_name LLaMA-3.1-8B-LBA-LoRA-ZeroShot\
    --context_window 2048 \
    --lora_rank 32 \
    --lora_alpha 16 \
    --epochs 5 \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --checkpointing_ratio 0.2 \
    --fp16 \
    --wandb_run_name $RunID \
    > ./Logs/$RunID.log 2>&1 &

$ nohup python -u -m LLaMA3_1.fine_tuning.lora_train_llama.py > ./Logs/FineTuning.log 2>&1 &
```