# LBA-DramaQG 2023 version

## Question Generation
- 불확실성에 대한 질문을 생성하는 모델입니다.

### How to use
  1. 다음과 같이 clone을 해주세요.
  ```
  git clone https://github.com/gminipark/LBA-DramaQG.git
  pip install -r requirements.txt
  ```
  
2. DramaQA 이미지 데이터셋을 다운로드 받아서 아래와 같이 디렉토리를  준비해주세요,
``` 
LBA-DramaQG/
    2022
    2023/
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
        blip2-flan-t5-xxl/
                

```
3. Generation
```
python run_inference.py --cache_dir="./blip2-flan-t5-xxl" --image_dir="./data/AnotherMissOh/AnotherMissOh_images/"
```

3-1. Diverse Question Generation
- prompt_type과 decoding_strategy를 통해 같은 입력에서 다양한 질문 생성가능
 - prompt_type = ["0", "1", "2"]
 - decoding_strategy = ["greedy", "beam", "constrastive", "diverse", "sample"]

```
python run_inference.py --cache_dir="./blip2-flan-t5-xxl" --image_dir="./data/AnotherMissOh/AnotherMissOh_images/" --prompt_type=2
```

## Input example 
```
[
    {"question" : "Why did Dokyung go to the old man?",
     "uncertain_information" : [{"['old','man']" : False}],
     "vid" : "AnotherMissOh17_001_0000"
    },
]
```
## Output example
```
['What does the old man look like?']
```

 ### Contact
	  - Gyu-Min Park (pgm1219@khu.ac.kr)
