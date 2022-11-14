
# LBA-DramaQG

## Question Generation
- 불확실성에 대한 질문을 생성하는 모델입니다.

### How to use
  1. 다음과 같이 clone을 해주세요.
  ```
  git clone https://github.com/gminipark/LBA-DramaQG.git
  pip install -r requirements.txt
  ```
  
2. DramaQA 데이터셋을 다운로드 받아서 아래와 같이 디렉토리를  준비해주세요,
``` 
LBA-DramaQG/
	dataset.py
	dataset_post.py
	train.py
	train_post.py
	generate.py
	generate_LBA.py
	preprocess.py   
	T5.py 
	requirements.txt

	DramaQA/
	      AnotherMissOhQA_test_set.json  
	      AnotherMissOhQA_train_set.json  
	      AnotherMissOhQA_val_set.json  
	      AnotherMissOh_script.json
```

3. QG 학습 데이터셋 전처리
```
python preprocess.py --directory_path DramaQA
```
---
```
	DramaQG/
		post-training.json
		fine-tuning.json
```
4. Post-training
```
python post-train.py
```
5. Fine-tuning
```
python train.py
```
6. Generation
```
python generate_LBA.py
```

### Fine-tuned 모델 다운로드
-  [link](https://drive.google.com/drive/folders/1M7gmjaoY8edl61J-BhbmJzPOLegwsT9s?usp=share_link)

 ### Contact
	  - Gyu-Min Park (pgm1219@khu.ac.kr)
	  - Seong-Eun Hong (zen152@khu.ac.kr)
