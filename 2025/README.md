# LBA-Relation Extraction
### 1. Clone repo 
```bash
git clone https://github.com/gminipark/LBA-DramaQG.git
```

### 2. Directory Structure
- .env 파일에 GPT API 작성
-  Data에 해당하는 디렉토리는 직접 생성해야함
```bash
.
└── LBA-DramaQG
    ├── 2022
    ├── 2023
    ├── 2024
    └── 2025
        ├── relation_extraction
        │   ├── .env
        │   ├── Data
        │   │   └── TACRED
        │   │        ├── dev.json
        │   │        ├── test.json
        │   │        └── train.json
        │   ├── run
        │   │   ├── output
        │   │   ├── classification.py
        │   │   ├── ...
        │   │   └── utils_for_tagging.py
        │   └── eval
        │       ├── after_refine_metric.py
        │       ├── ...
        │       └── classification_confusion.py
        └── ...
```
