import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    TrainingArguments,
    TextStreamer,
)
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    TaskType,
)
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from accelerate import PartialState, prepare_pippy
# device_string = PartialState().process_index  # For Model Parallel Fine-Tuning
# local_rank = int(os.environ["LOCAL_RANK"])

PAD_TOKEN = "<|pad|>"
os.environ["WANDB_PROJECT"] = "LBA-Text2Data"

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_text_field", type=str, default="Text_FewZero", help="column name for Specific Text Data from Training Dataset; defaults to Text_FewShot")
parser.add_argument("--base_model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Base Model to Train on (HF Model Path; defaults to meta-llama/Meta-llama-3.1-8B-Instruct)")
parser.add_argument("--new_model_name", type=str, default="LLaMA-3.1-8B-LBA-LoRA", help="New Model Name (without Path Specifier; defaults to LLaMA-3.1-8B-LBA-LoRA)")
parser.add_argument(
    "--context_window",
    type=int,
    default=3072,
    help="Context Window Size for LLaMA3.1 (defaults to 3072)",
)
parser.add_argument("--lora_rank", type=int, default=16, help="Rank for LoRA (defaults to 16)")
parser.add_argument("--lora_alpha", type=int, default=16, help="Alpha for LoRA (defaults to 16)")
parser.add_argument("--epochs", type=int, default=1, help="# of Epochs (defaults to 1)")
parser.add_argument("--per_device_batch_size", type=int, default=1, help="# of Batches per GPU (defaults to 1)")
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=4,
    help="# of examples to accumulate gradients before taking a step (defaults to 1)",
)
parser.add_argument(
    "--checkpointing_ratio",
    type=float,
    default=0.25,
    help="Percentage of Epochs to be Completed Before a Model Saving Happens (defaults to 0.25)",
)
parser.add_argument("--fp16", action="store_true", help="whether or not to use FP16")
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default="base",
    help="Wandb Logging Name for this Training Run",
)
parser.add_argument(
    "--test",
    action="store_true",
    help="use very small dataset(~100) to validate the fine-tuning process",
)
args = parser.parse_args()


# Load_Dataset
uncompressed_dir = "./Data/DramaQA_KG_Processed/TrainingData/uncompressed/"
train_dir = os.path.join(uncompressed_dir, "train_uncomp.json")
val_dir = os.path.join(uncompressed_dir, "val_uncomp.json")
test_dir = os.path.join(uncompressed_dir, "test_uncomp.json")
print("Loading Dataset")
dataset = load_dataset(
    "json",
    data_files={"train": train_dir, "validation": val_dir, "test": test_dir},
)
print("Done"); print(dataset)

base_model_path = args.base_model_path
new_model_name = args.new_model_name

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_special_tokens({"pad_token":PAD_TOKEN})  # pad token으로 eos_token 사용하는 대신 추가해줌
tokenizer.padding_side = "right"  # left -> right 수정함


response_template = "<|end_header_id|>"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)  # Generation Part 제외한 Instruction, FewShot Example 부분은 -100으로 마스킹하여 파인튜닝 성능개선


# 1: Initialize the Base Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto',
    # device_map={'':device_string},
    # load_in_4bit=True,
)

# 2: Lora Configuration to define LoRA-specific Parameters
lora_config = LoraConfig(
    lora_alpha=args.lora_alpha,  # LoRA Scaling Factor
    lora_dropout=0.05,  # 
    r=args.lora_rank,  # the Rank of the Update Matrices (int) -> Lower Rank==Smaller Update Matrices (w. Fewer Parameters)
    use_rslora=True,  # Rank-Stablized Scaling for LoRA
    bias='none',
    init_lora_weights="gaussian",  # initialize weight A with Gaussian Distribution (weight B is still Zero-Initialized)
    task_type=TaskType.CAUSAL_LM,
)

# 3: Wrap the Base Model with get_peft_model() to get a trainable PeftModel
peft_model = get_peft_model(base_model, lora_config)

# 4: Train the PeftModel (same way as training base model)
sft_config = SFTConfig(
    output_dir="./LLaMA3_1/fine_tuning/fine_tuned_models",
    dataset_text_field=args.dataset_text_field,
    max_seq_length=args.context_window,
    num_train_epochs=args.epochs, 
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    per_device_train_batch_size=args.per_device_batch_size, 
    per_device_eval_batch_size=args.per_device_batch_size, 
    fp16=args.fp16, 
    dataset_kwargs={
        "add_special_tokens":False,
        "append_concat_token":False,
    },
    save_strategy="steps",
    save_steps=args.checkpointing_ratio,
    evaluation_strategy="steps",
    eval_steps=args.checkpointing_ratio,
    report_to="wandb",
    run_name=args.wandb_run_name,
    logging_steps=1,
)
trainer = SFTTrainer(
    peft_model,
    args=sft_config,
    # pert_config=lora_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

peft_model.print_trainable_parameters()

# 5: Start Training
trainer.train()

trainer.save_model()