# LBA-DramaQG 2022 version

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
1. Generation
```
python run_inference.py --cache_dir="./blip2-flan-t5-xxl" --image_dir="./data/AnotherMissOh/AnotherMissOh_images/"
```

## Input example 
```
[
    {"question" : "Why did Dokyung go to the old man?",
     "uncertain_information" : "man",
     "vid" : "AnotherMissOh17_001_0000"
    },
]
```
## Output example
```
["What is the man's name?"]
```

 ### Contact
	  - Gyu-Min Park (pgm1219@khu.ac.kr)
