# LBA-DramaQG 2024 version

## Question Generation

- 불확실성에 대한 질문을 생성하는 모델입니다.

### How to use

1. 다음과 같이 clone을 해주세요.

```
git clone https://github.com/gminipark/LBA-DramaQG.git

```

2. 필요한 라이브러리 설치와 모델을 다운로드 받습니다.

```
bash init.sh

```
링크: [link](http://gofile.me/5YLyZ/MgHW975A)

3. DramaQA 이미지 데이터셋을 다운로드 받아서 아래와 같이 디렉토리를 준비해주세요,

```
LBA-DramaQG/
    2022
    2023
    2024/
        dataset.py
	    image_utils.py
        utils.py
	    run_inference.py
	    generate_LBA.py
	    requirements.txt
	    data/
	        AnotherMissOhQA
  		    /AnotherMissOh_images
		        /AnotherMissOh01
			        /001
                        ...
                        	...

```

4. Generation

```
CUDA_VISIBLE_DEVICES=0 python run_inference.py --model_path=llava-checkpoint --video_dir="./data/AnotherMissOh/AnotherMissOh_images/" --input_file=input_sample.json --output_dir=./ --output_name=output_sample.json                       
```

- input_path: 입력 json 파일 경로
- output_path: 출력 json 파일 경로

4-1. Diverse Question Generation

- decoding_strategy를 통해 같은 입력에서 다양한 질문 생성가능
- decoding_strategy = ["greedy", "sample"]

```
CUDA_VISIBLE_DEVICES=0 python run_inference.py --model_path=llava-checkpoint --video_dir="./data/AnotherMissOh/AnotherMissOh_images/" --input_file=input_sample.json --output_dir=./ --output_name=output_sample.json --decoding_strategy sample
```

## Input json example

```
[
    {
        "qid": 3205,
        "question": "How is the relationship between Haeyoung1 and Dokyung when the two hug and kiss each other?",
        "answerable": "no",
        "reasoning": "Dokyung",
        "vid": "AnotherMissOh14_001_0000"
    },
    ...
]
```

## Output json example

```
[
    {
        "qid": 3205,
        "question": "How is the relationship between Haeyoung1 and Dokyung when the two hug and kiss each other?",
        "answerable": "no",
        "reasoning": "Dokyung",
        "vid": "AnotherMissOh14_001_0000",
        "pred": What is Dokyung doing?
    },
    ...
]
```

### Contact

      - Gyu-Min Park (pgm1219@khu.ac.kr)
