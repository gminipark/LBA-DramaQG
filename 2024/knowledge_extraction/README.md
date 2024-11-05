# LBA - Knowledge Extraction (Text2Data) 2024 version

## Setting
### 1. Clone Repo
```bash
git clone https://github.com/gminipark/LBA-DramaQG.git
```
### 2. Install Dependencies
코드 실행에 필요한 관련 라이브러리를 설치
```bash
$ pip install -r requirements.txt
```
### 3. Directory Structure
아래와 같은 디렉토리 구축
(Data에 해당하는 디렉토리는 직접 생성해야함)
```bash
.
└── LBA-DramaQG
    ├── 2022
    ├── 2023
    └── 2024
        ├── knowledge_extraction
        │   ├── Data
        │   │   ├── DramaQA_KG
        │   │   ├── DramaQA_KG_Processed
        │   │   └── FewShotDemo
        │   ├── LLaMA_3_1
        │   │   └── fine_tuning
        │   └── Test
        └── ...
```
### 4. Dataset Save
[파일 저장된 깃헙 링크](https://github.com/wjcldply/LBA-Text2Data-Public.git)에서 `Data` 디렉토리 내의 `DramaQA_KG`, `FewShotDemo` 파일들을 다운로드하고 동일 경로에 저장해야 함

### 5. SETUP & LOGIN
- (터미널 내에서) HF🤗 & WANDB & OpenAI 로그인

## Training / Testing

### Data-Building
- Text2Data 데이터셋 / 스키마셋 생성
    ```bash
    $ python -m LLaMA3_1.Test.kg_json_rows_prep.py
    ```
- 테스트용 랜덤샘플 생성
    ```bash
    $ python -m LLaMA3_1.Test.json_random_picker.py
    ```
- Fine-Tuning Dataset 생성
    ```bash
    $ python -m LLaMA3_1.fine_tuning.train_data_build
    ```

### Fine-Tuning
```bash
$ export RunID=WANDB_RUNID_TO_USE

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

### Inference (BackBone: HF Transformers; LLaMA3.1-8B)
```bash
$ python -m Test.eval_week9.py
```