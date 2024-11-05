import re
import sys
import time
import json
import torch
import openai
from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
logging.set_verbosity_error()
from tqdm import tqdm
from pprint import pprint

from tester_utils import parse_assistant

class Tester_v6:
    """for accuracy-based (exact matching) comparison"""
    def __init__(self, list_of_sentences, demonstrations=None, schemas=None, multiTurnQA=False, batch_size=1, openai_api_key=None, openai_model='gpt-4o-mini', lora_adapter_path=None):
        """
        list_of_sentences: (list of strings) input text to conduct OIE on.
        demonstrations: (list of strings, defaults to None) few-shot demonstrations 
        schemas: (dict, defaults to None) schemas(objects, attributes, relations) extracted from the validation set
        """
        if openai_api_key is None:
            self.openai_api_key = None
            self.model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"  # left -> right 수정함
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                # load_in_8bit=True,  # Use 8-bit precision
                torch_dtype=torch.float16,  # Optional: can use float16 for inference calculations
                # device_map='cuda'
                device_map='auto'
            )
            if lora_adapter_path is not None:  # LoRA 어댑터 경로 전달 시
                self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
        else:
            self.openai_model = openai_model
            self.openai_api_key = openai_api_key
            openai.api_key = openai_api_key

        self.final_format_description = """
        Your responses must strictly adhere to the JSON formats provided below.
        Always avoid including any additional notes, explanations, or extra text outside the JSON structure.
        Make sure to find all triples and doubles you are confident about, but if you are not confident it's okay not to extract either one of them.
    
        Respond with the following JSON string format: 
        {
            "extracted_KGs": [
                {
                    "object": "",
                    "attribute": ""
                },
                {
                    "object": "",
                    "attribute": ""
                },
                {
                    "object1": "",
                    "relation": "",
                    "object2": ""
                },
                {
                    "object1": "",
                    "relation": "",
                    "object2": ""
                }
            ]
        }
        If a double or triple is incomplete or cannot be determined, do not include it in the response.
        The JSON must be valid and contain only the required fields.
        Make sure not to contain trailing comma at the last element in each lists.
        Make sure you have all the parenthesis, braces, and brackets correct so that json structure is preserved.
        If there's nothing to extract, just leave the value for the key "extracted_KGs" as empty list like such: [].
        """

        self.system_message = """
        You are an Knowledge Graph (KG) Extraction system designed to extract objects, relations and attributes from the given text, and create triples or doubles with them.
        Ensure all triples and doubles are complete and accurately reflect the information in the input text.""" + "\n" + self.final_format_description

        self.list_extract_format_description = """
        Your responses must strictly adhere to the formats provided below.

        Respond with the following comma-separated elements format: 
        element1, element2, element3, ...
        
        If there's nothing to extract, just respond as None format:
        None

        Always avoid including any additional notes, explanations, or extra text outside the provided structure.    
        Make sure not to contain trailing comma at the last element.
        You MUST Strictly follow the provided Schema. Don't Extract Elements Outside the Given Schema. Your Responses Must be Chosen from Provided Schema List. Don't Alter or Modify Element from Provided Schema List, and Output Each of them Exactly the way they were from the Given Schema List.
        """

        self.list_of_sentences = list_of_sentences
        self.demonstrations = demonstrations if demonstrations is not None else []  # few-shot demonstration이 None일 경우 []로(Zero-Shot), list of string일 경우 [strings]로(Few-Shot)
        self.schemas = schemas
        self.multiTurnQA = multiTurnQA
        self.batch_size = batch_size
        if self.multiTurnQA is False:  # Single Turn
            if self.schemas is not None:  # Single Turn -> 주어진 스키마가 존재하여 이를 활용해야 하는 경우
                self.system_message_schema_description = f"""\nFollowings are objects, attributes, and relations schemas defined in the current Knowledge-Base.
                objects: {schemas['objects']}
                attributes: {schemas['attributes']}
                relations: {schemas['relations']}
                Make sure you extract objects, attributes, and relations that already exist in the provided schemas.
                """
            self.batch_inference = self.batch_inference_SingleTurn
        else:  # Multi Turn QA
            self.batch_inference = self.batch_inference_MultiTurnQA

        self.verbose=True
        
    def generate_batch(self, mode=None, re_ae_ee=None):
        """
        mode : 프롬프트 배치 생성에 사용하는 설정값임. 프롬프팅 스타일에 따라 변화
            mode=None : 
            mode='RE' : Multi-Turn QA 스타일 프롬프팅을 위한 배치 프롬프트 제너레이션 진행, Relation Extraction 단계를 위한 프롬프트
            mode='AE' : Multi-Turn QA 스타일 프롬프팅을 위한 배치 프롬프트 제너레이션 진행, Attribute Extraction 단계를 위한 프롬프트
            mode='EE' : Multi-Turn QA 스타일 프롬프팅을 위한 배치 프롬프트 제너레이션 진행, Entity Extraction 단계를 위한 프롬프트
            mode='FINAL' : Multi-Turn QA 스타일 프롬프팅을 위한 배치 프롬프트 제너레이션 진행, 앞선 세 단계에서 추출한 Relations, Attributes, Entities의 리스트들을 담은 List of List [[Relation 1, Relation 2, ...], [Attribute 1, Attribute 2, ...], [Entity 1, Entity 2, ...]]를 re_ae_ee 파라메터로 전달받아 사용함
        re_ae_ee : mode='FINAL'에 사용 (list of list; [[Relation 1, Relation 2, ...], [Attribute 1, Attribute 2, ...], [Entity 1, Entity 2, ...]])
        """
        batch_prompt = []
        if mode is None:  # Single Turn인 경우
            for sentence in self.list_of_sentences:
                fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
                for demonstration in self.demonstrations:  # Few-Shot Demonstrations (added before the target sentence)
                    fs_prompt.append({"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {demonstration['input']}"})
                    fs_prompt.append({"role": "assistant", "content": json.dumps({"extracted_KGs": demonstration["output"]["gold"]}, indent=4)})
                final_prompt=[{"role": "user", "content": f"Extract list of KG(Knowledge Graph) doubles and triples from the following sentence: {sentence}"}]  # Add the target sentence for inference
                if self.openai_api_key is None:
                    formatted_prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": self.system_message+"\n"+self.system_message_schema_description}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=True)  # Use tokenizer's chat template application to handle system message and tokenization
                else:
                    formatted_prompt = [{"role": "system", "content": self.system_message+"\n"+self.system_message_schema_description}] + fs_prompt + final_prompt
                batch_prompt.append(formatted_prompt)  # Append the formatted prompt to batch
        elif mode == 'RE':
            for sentence in self.list_of_sentences:
                fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
                for demonstration in self.demonstrations:  # Few-Shot Demonstrations (added before the target sentence)
                    fs_prompt.append({"role": "user", "content": f"The given sentence is {demonstration['input']['sentence']}\n\nList of given relations: {demonstration['input']['relations_schema']}\n\nWhat relations in the given list are included in the given sentence?\nIf not present, respond with and only with: None\nIf present, respond as a listing of present relations separated with comma(,) (e.g. relation 1, relation 2, ...) Don't Extract More Items than What's Actually In the Given Sentence."})
                    fs_prompt.append({"role": "assistant", "content": ', '.join(demonstration["output"]["relations"])})
                final_prompt=[{"role": "user", "content": f"The given sentence is {sentence}\n\nList of given relations: {self.schemas['relations']}\n\nWhat relations in the given list are included in the given sentence?\nIf not present, respond with and only with: None\nIf present, respond as a listing of present relations separated with comma(,) (e.g. relation 1, relation 2, ...) Don't Extract More Items than What's Actually In the Given Sentence."}]  # Add the target sentence for inference
                if self.openai_api_key is None:
                    formatted_prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": self.list_extract_format_description}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=True)  # Use tokenizer's chat template application to handle system message and tokenization
                else:
                    formatted_prompt = [{"role": "system", "content": self.list_extract_format_description}] + fs_prompt + final_prompt
                batch_prompt.append(formatted_prompt)  # Append the formatted prompt to batch                
        elif mode == 'AE':
            for sentence in self.list_of_sentences:
                fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
                for demonstration in self.demonstrations:  # Few-Shot Demonstrations (added before the target sentence)
                    fs_prompt.append({"role": "user", "content": f"The given sentence is {demonstration['input']['sentence']}\n\nList of given attributes: {demonstration['input']['attributes_schema']}\n\nWhat attributes in the given list are included in the given sentence?\nIf not present, respond with and only with: None\nIf present, respond as a listing of present attributes separated with comma(,) (e.g. attribute 1, attribute 2, ...) Don't Extract More Items than What's Actually In the Given Sentence."})
                    fs_prompt.append({"role": "assistant", "content": ', '.join(demonstration["output"]["attributes"])})
                final_prompt=[{"role": "user", "content": f"The given sentence is {sentence}\n\nList of given attributes: {self.schemas['attributes']}\n\nWhat attributes in the given list are included in the given sentence?\nIf not present, respond with and only with: None\nIf present, respond as a listing of present attributes separated with comma(,) (e.g. attribute 1, attribute 2, ...) Don't Extract More Items than What's Actually In the Given Sentence."}]  # Add the target sentence for inference
                if self.openai_api_key is None:
                    formatted_prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": self.list_extract_format_description}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=True)  # Use tokenizer's chat template application to handle system message and tokenization
                else:
                    formatted_prompt = [{"role": "system", "content": self.list_extract_format_description}] + fs_prompt + final_prompt
                batch_prompt.append(formatted_prompt)  # Append the formatted prompt to batch
        elif mode == 'EE':
            for sentence in self.list_of_sentences:
                fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
                for demonstration in self.demonstrations:  # Few-Shot Demonstrations (added before the target sentence)
                    fs_prompt.append({"role": "user", "content": f"The given sentence is {demonstration['input']['sentence']}\n\nList of given entities: {demonstration['input']['entities_schema']}\n\nWhat entities in the given list are included in the given sentence?\nIf not present, respond with and only with: None\nIf present, respond as a listing of present entities separated with comma(,) (e.g. entity 1, entity 2, ...) Don't Extract More Items than What's Actually In the Given Sentence."})
                    fs_prompt.append({"role": "assistant", "content": ', '.join(demonstration["output"]["entities"])})
                final_prompt=[{"role": "user", "content": f"The given sentence is {sentence}\n\nList of given entities: {self.schemas['objects']}\n\nWhat entities in the given list are included in the given sentence?\nIf not present, respond with and only with: None\nIf present, respond as a listing of present entities separated with comma(,) (e.g. entity 1, entity 2, ...) Don't Extract More Items than What's Actually In the Given Sentence."}]  # Add the target sentence for inference
                if self.openai_api_key is None:
                    formatted_prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": self.list_extract_format_description}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=True)  # Use tokenizer's chat template application to handle system message and tokenization
                else:
                    formatted_prompt = [{"role": "system", "content": self.list_extract_format_description}] + fs_prompt + final_prompt
                batch_prompt.append(formatted_prompt)  # Append the formatted prompt to batch
        elif mode == 'FINAL':
            assert re_ae_ee is not None and len(re_ae_ee[0]) == len(re_ae_ee[1]) == len(re_ae_ee[2]) == len(self.list_of_sentences)
            for index, sentence in enumerate(self.list_of_sentences):
                # print(f'Extracted Relations({len(re_ae_ee[0][index])}):{re_ae_ee[0][index]}')
                # print(f'Extracted Attributes({len(re_ae_ee[1][index])}):{re_ae_ee[1][index]}')
                # print(f'Extracted Entities({len(re_ae_ee[2][index])}):{re_ae_ee[2][index]}')
                fs_prompt = []  # Initialize the prompt list with few-shot demonstrations
                for demonstration in self.demonstrations:  # Few-Shot Demonstrations (added before the target sentence)
                    fs_prompt.append({"role": "user", "content": f"The given sentence is {demonstration['input']['sentence']}\n\nList of extracted relations: {demonstration['output']['relations']}\nList of extracted attributes: {demonstration['output']['attributes']}\nList of extracted entities: {demonstration['output']['entities']}\n\nCompare the given Sentence with extracted Relations, Attributes, and Entities to Create Triples(Object1, Relation, Object2) and Doubles(Object, Attribute). Don't Extract More Items than What's Actually In the Given Sentence or Schema."})  #  Present them in the format of a list of KG(Knowledge Graph) doubles and triples.
                    fs_prompt.append({"role": "assistant", "content": json.dumps({"extracted_KGs": demonstration["output"]["gold"]}, indent=4)})
                final_prompt=[{"role": "user", "content": f"The given sentence is {sentence}\n\nList of extracted relations: {re_ae_ee[0][index]}\nList of extracted attributes: {re_ae_ee[1][index]}\nList of extracted entities: {re_ae_ee[2][index]}\n\nCompare the given Sentence with extracted Relations, Attributes, and Entities to Create Triples(Object1, Relation, Object2) and Doubles(Object, Attribute). Don't Extract More Items than What's Actually In the Given Sentence or Schema."}]  # Present them in the format of a list of KG(Knowledge Graph) doubles and triples. # Add the target sentence for inference
                if self.openai_api_key is None:
                    formatted_prompt = self.tokenizer.apply_chat_template([{"role": "system", "content": self.final_format_description}] + fs_prompt + final_prompt, tokenize=False, add_generation_prompt=True)  # Use tokenizer's chat template application to handle system message and tokenization
                else:
                    formatted_prompt = [{"role": "system", "content": self.final_format_description}] + fs_prompt + final_prompt
                batch_prompt.append(formatted_prompt)  # Append the formatted prompt to batch
        else:
            pass

        return batch_prompt
    

    def batch_inference_SingleTurn(self, max_retries=2, retry_delay=1):
        batch_size = self.batch_size
        all_prompts = self.generate_batch()
        responses = []

        # Divide into Smaller Batch Sizes
        for i in tqdm(range(0, len(all_prompts), batch_size), desc="Processing batches"):
            batch_prompt = all_prompts[i:i+batch_size]

            if self.openai_api_key is None:
                inputs = self.tokenizer(
                    batch_prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(self.model.device)
                # allocated_memory = torch.cuda.memory_allocated(device=self.model.device)
                # allocated_memory_mb = allocated_memory / (1024 ** 2)  # Convert bytes to MB
                # print(f"GPU RAM Allocation after tokenization / before generation: {allocated_memory_mb:.2f} MB")

                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_new_tokens=512,
                    eos_token_id=self.terminators,
                    do_sample=True,
                    temperature=0.8,  # Temp가 높을수록 OIE 출력 format에 더 잘 부합하는 결과를 생성하는 점은 장점이나, paraphrasing 등이 요구되는 어려운 추출의 경우 성능이 하락할 것으로 예상됨 (further experiments needed)
                    top_p=0.9,
                )

                for _, output in enumerate(outputs):
                    response = output[inputs['input_ids'].shape[-1]:]
                    response_decoded = self.tokenizer.decode(response, skip_special_tokens=True)
                    response_decoded = response_decoded.replace("'", '"')
                    # print(f'\ndecoded response: {response_decoded}({type(response_decoded)})')
                    if response_decoded[:9] == "assistant":
                        response_decoded = parse_assistant(response_decoded)
                    print(f"\nDecoded Response: {response_decoded}")
                    
                    # Retry mechanism for JSON Decoding Error
                    retries = 0
                    while retries < max_retries:
                        try:
                            # Attempt to load the response as JSON or List
                            # if type(eval(response_decoded)) is list:  # LoRA 모델의 경우 리스트만 출력하는 경향 발견
                            #     response_json_KGs = eval(response_decoded)
                            # else: # 프롬프트로 요구한 바와 같이 딕셔너리 출력한 경우
                            response_json_object = json.loads(response_decoded)  # Convert string to JSON object
                            if isinstance(response_json_object, list):
                                response_json_KGs = response_json_object
                            else:
                                response_json_KGs = response_json_object["extracted_KGs"]
                            responses.append(response_json_KGs)
                            break  # 정상적으로 json decoding 되는 정상답변이 생성된 경우 바로 루프 탈출
                        except json.JSONDecodeError as e:
                            retries += 1
                            print(f"JSON Decode Error (Retry {retries}/{max_retries}): {e}"); sys.stdout.flush()
                            print(f"Problematic response: {response_decoded}"); sys.stdout.flush()
                            # Retry Inferencing until correct json structure is generated
                            if retries < max_retries:
                                print(f'Re-inferencing...'); sys.stdout.flush()
                                time.sleep(retry_delay)
                                new_outputs = self.model.generate(
                                    inputs['input_ids'],
                                    attention_mask=inputs['attention_mask'],
                                    max_new_tokens=512,
                                    eos_token_id=self.terminators,
                                    do_sample=True,
                                    temperature=0.8,  # Temp가 높을수록 OIE 출력 format에 더 잘 부합하는 결과를 생성하는 점은 장점이나, paraphrasing 등이 요구되는 어려운 추출의 경우 성능이 하락할 것으로 예상됨 (further experiments needed)
                                    top_p=0.9,
                                )
                                # 새로 생성한 response를 기존 변수에 대입 (덮어쓰기)
                                response = new_outputs[inputs['input_ids'].shape[-1]:]
                                response_decoded = self.tokenizer.decode(response, skip_special_tokens=True)
                            else:  # 지정된 최대 재시도 횟수 넘길 경우 그냥 스킵
                                print(f"Failed to get valid JSON after {max_retries} attempts. Skipping response."); sys.stdout.flush()
            else:   #OPENAI API
                # Call the OpenAI API for each batch of prompts
                for prompt in batch_prompt:
                    retries = 0
                    while retries < max_retries:
                        try:
                            # Perform GPT-4 Inference
                            response = openai.ChatCompletion.create(
                                model=self.openai_model,
                                messages=prompt,
                                max_tokens=512,
                                temperature=0.8,
                                top_p=0.9
                            )
                            response_text = response['choices'][0]['message']['content']

                            # Try to parse the response as JSON
                            response_json_object = json.loads(response_text)  # Convert string to JSON object
                            if isinstance(response_json_object, list):
                                response_json_KGs = response_json_object
                            else:
                                response_json_KGs = response_json_object["extracted_KGs"]
                            responses.append(response_json_KGs)
                            break  # 정상적으로 json decoding 되는 정상답변이 생성된 경우 바로 루프 탈출

                        except (json.JSONDecodeError, KeyError) as e:
                            retries += 1
                            print(f"JSON Decode Error (Retry {retries}/{max_retries}): {e}")
                            print(f"Problematic response: {response_text}")
                            if retries < max_retries:
                                print(f"Re-inferencing...")
                                time.sleep(60)
                            else:
                                print(f"Failed to get valid JSON after {max_retries} attempts. Skipping response.")
                                responses.append(None)  # or handle it differently if needed

        return responses  # list of list of dicts
    
    def batch_inference_MultiTurnQA(self, max_retries=2, retry_delay=1, extraction_count_limit=10):
        """
        extraction_count_limit : (int) RE, AE, EE 각 단계에서 추출하는 element가 비정상적으로 길 경우 지정된 수 이상은 버리기 위해 설정하는 값임
        """
        batch_size = self.batch_size
        prompts_RE = self.generate_batch(mode='RE')  # batched Prompt for Turn 1: RE (Sentence(String)+Relation Schema(List) -> Relation Types (present in Sentence, List))
        responses_RE = []  # Turn 1에서 추출한 (각 sentence별로 감지된) Relation Types from Relation Schemas가 담긴 list들을 담고 있는, list of list 변수

        for i in tqdm(range(0, len(prompts_RE), batch_size), desc="Processing Batches for Turn 1 (RE)"):
            # Divide into Smaller Batch Sizes
            batch_prompt = prompts_RE[i:i+batch_size]

            # Turn 1: RE (Sentence(String)+Relation Schema(List) -> Relation Types (present in Sentence, List))
            if self.openai_api_key is None:
                inputs_RE = self.tokenizer(batch_prompt, return_tensors="pt", padding=True, truncation=True,).to(self.model.device)
                outputs_RE = self.model.generate(inputs_RE['input_ids'], attention_mask=inputs_RE['attention_mask'], max_new_tokens=512, eos_token_id=self.terminators, do_sample=True, temperature=0.8, top_p=0.9,)
                # 요구된 포멧(List)에 맞는 답변이라면 Turn 3: EE에서 사용하기 위한 Relation Type List로 저장하고, 만약 요구된 포멧을 따르지 않는 답변이 생성됐을 경우 생성 재시도
                for _, output_RE in enumerate(outputs_RE):
                    response_RE = output_RE[inputs_RE['input_ids'].shape[-1]:]
                    response_decoded_RE = self.tokenizer.decode(response_RE, skip_special_tokens=True)  # 이상적으론 List 형태의 String임
                    # Retry mechanism
                    retries = 0
                    while retries < max_retries:
                        try:
                            # Attempt to load the response as List
                            response_list_object = [] if response_decoded_RE == 'None' else [item.strip().rstrip('.') for item in response_decoded_RE.split(",")]
                            response_list_object = list(set(response_list_object))  # Duplicate 제거
                            filtered_RE = [item for item in response_list_object if item != '' and item in self.schemas['relations']]  # Filtering
                            responses_RE.append(filtered_RE[:extraction_count_limit])
                            break  # 정상적으로 list화되는 정상답변이 생성된 경우 바로 루프 탈출
                        except (SyntaxError, NameError, TypeError, ValueError, AttributeError) as e:
                            retries += 1
                            print(f"List Eval Error (Retry {retries}/{max_retries}): {e}"); sys.stdout.flush()
                            print(f"Problematic response: {response_decoded_RE}"); sys.stdout.flush()                        
                            if retries < max_retries:  # Retry Inferencing until correct list structure is generated
                                print(f'Re-inferencing...'); sys.stdout.flush()
                                time.sleep(retry_delay)
                                new_outputs_RE = self.model.generate(inputs_RE['input_ids'], attention_mask=inputs_RE['attention_mask'], max_new_tokens=512, eos_token_id=self.terminators, do_sample=True, temperature=0.8, top_p=0.9,)
                                # 새로 생성한 response를 기존 변수에 대입 (덮어쓰기)
                                response_RE = new_outputs_RE[inputs_RE['input_ids'].shape[-1]:]
                                response_decoded_RE = self.tokenizer.decode(response_RE, skip_special_tokens=True)
                            else:
                                print(f"Failed to get valid LIST of RE Types after {max_retries} attempts. Skipping Response."); sys.stdout.flush()
                                responses_RE.append([])  # 지정된 최대 재시도 횟수 넘길 경우 그냥 스킵하고 빈 리스트 저장
            else:  # OPENAI API
                batch_responses_RE = []
                for prompt in batch_prompt:
                    retries = 0
                    while retries < max_retries:
                        try:
                            response = openai.ChatCompletion.create(
                                model=self.openai_model,
                                messages=prompt,
                                max_tokens=512,
                                temperature=0.8,
                                top_p=0.9,
                            )

                            response_decoded_RE = response['choices'][0]['message']['content']
                            # Attempt to load the response as List
                            response_list_object = [] if response_decoded_RE == 'None' else [item.strip().rstrip('.') for item in response_decoded_RE.split(",")]
                            response_list_object = list(set(response_list_object))  # Remove duplicates
                            filtered_RE = [item for item in response_list_object if item != '' and item in self.schemas['relations']]  # Filtering
                            batch_responses_RE.append(filtered_RE[:extraction_count_limit])
                            break  # Exit loop if successful
                        except (openai.error.OpenAIError, SyntaxError, NameError, TypeError, ValueError, AttributeError) as e:
                            retries += 1
                            print(f"OpenAI API Error / List Eval Error (Retry {retries}/{max_retries}): {e}")
                            print(f"Problematic response: {response_decoded_RE}")
                            if retries < max_retries:
                                print(f'Re-inferencing...')
                                time.sleep(60)
                            else:
                                print(f"Failed to get valid LIST of RE Types after {max_retries} attempts. Skipping Response.")
                                batch_responses_RE.append([])  # Append an empty list if failed
                responses_RE.extend(batch_responses_RE)  # Add this batch's responses to the main list
        if self.verbose is True:
            print(f'Extracted Relations: {responses_RE}')

        prompts_AE = self.generate_batch(mode='AE')  # batched Prompt for Turn 2: AE (Sentence(String)+Attribute Schema(List) -> Attribute Types (present in Sentence, List))
        responses_AE = []  # Turn 2에서 추출한 (각 sentence별로 감지된) Attribute Types from Attribute Schemas가 담긴 list들을 담고 있는, list of list 변수

        for i in tqdm(range(0, len(prompts_AE), batch_size), desc="Processing Batches for Turn 2 (AE)"):
            # Divide into Smaller Batch Sizes
            batch_prompt = prompts_AE[i:i+batch_size]

            if self.openai_api_key is None:
                # Turn 1: AE (Sentence(String)+Attribute Schema(List) -> Attribute Types (present in Sentence, List))
                inputs_AE = self.tokenizer(batch_prompt, return_tensors="pt", padding=True, truncation=True,).to(self.model.device)
                # allocated_memory = torch.cuda.memory_allocated(device=self.model.device)
                # allocated_memory_mb = allocated_memory / (1024 ** 2)  # Convert bytes to MB
                # print(f"GPU RAM Allocation after tokenization / before generation: {allocated_memory_mb:.2f} MB")
                outputs_AE = self.model.generate(inputs_AE['input_ids'], attention_mask=inputs_AE['attention_mask'], max_new_tokens=512, eos_token_id=self.terminators, do_sample=True, temperature=0.8, top_p=0.9,)

                # 요구된 포멧(List)에 맞는 답변이라면 Turn 3: EE에서 사용하기 위한 Attribute Type List로 저장하고, 만약 요구된 포멧을 따르지 않는 답변이 생성됐을 경우 생성 재시도
                for _, output_AE in enumerate(outputs_AE):
                    response_AE = output_AE[inputs_AE['input_ids'].shape[-1]:]
                    response_decoded_AE = self.tokenizer.decode(response_AE, skip_special_tokens=True)  # 이상적으론 List 형태의 String임
                    # Retry mechanism
                    retries = 0
                    while retries < max_retries:
                        try:
                            # Attempt to load the response as List
                            response_list_object = [] if response_decoded_AE == 'None' else [item.strip().rstrip('.') for item in response_decoded_AE.split(",")]
                            response_list_object = list(set(response_list_object))  # Duplicate 제거
                            filtered_AE = [item for item in response_list_object if item != '' and item in self.schemas['attributes']]  # Filtering
                            responses_AE.append(filtered_AE[:extraction_count_limit])
                            break  # 정상적으로 list화되는 정상답변이 생성된 경우 바로 루프 탈출
                        except (SyntaxError, NameError, TypeError, ValueError) as e:
                            retries += 1
                            print(f"List Eval Error (Retry {retries}/{max_retries}): {e}"); sys.stdout.flush()
                            print(f"Problematic response: {response_decoded_AE}"); sys.stdout.flush()                        
                            if retries < max_retries:  # Retry Inferencing until correct list structure is generated
                                print(f'Re-inferencing...'); sys.stdout.flush()
                                time.sleep(retry_delay)
                                new_outputs_AE = self.model.generate(inputs_AE['input_ids'], attention_mask=inputs_AE['attention_mask'], max_new_tokens=512, eos_token_id=self.terminators, do_sample=True, temperature=0.8, top_p=0.9,)
                                # 새로 생성한 response를 기존 변수에 대입 (덮어쓰기)
                                response_AE = new_outputs_AE[inputs_AE['input_ids'].shape[-1]:]
                                response_decoded_AE = self.tokenizer.decode(response_AE, skip_special_tokens=True)
                            else:
                                print(f"Failed to get valid LIST of AE Types after {max_retries} attempts. Skipping Response."); sys.stdout.flush()
                                responses_AE.append([])  # 지정된 최대 재시도 횟수 넘길 경우 그냥 스킵하고 빈 리스트 저장
            else:  # OPENAI API
                batch_responses_AE = []
                for prompt in batch_prompt:
                    retries = 0
                    while retries < max_retries:
                        try:
                            response = openai.ChatCompletion.create(
                                model=self.openai_model,
                                messages=prompt,
                                max_tokens=512,
                                temperature=0.8,
                                top_p=0.9,
                            )

                            response_decoded_AE = response['choices'][0]['message']['content']
                            # Attempt to load the response as List
                            response_list_object = [] if response_decoded_AE == 'None' else [item.strip().rstrip('.') for item in response_decoded_AE.split(",")]
                            response_list_object = list(set(response_list_object))  # Remove duplicates
                            filtered_AE = [item for item in response_list_object if item != '' and item in self.schemas['attributes']]  # Filtering
                            batch_responses_AE.append(filtered_AE[:extraction_count_limit])
                            break  # Exit loop if successful
                        except (openai.error.OpenAIError, SyntaxError, NameError, TypeError, ValueError, AttributeError) as e:
                            retries += 1
                            print(f"OpenAI API Error / List Eval Error (Retry {retries}/{max_retries}): {e}")
                            print(f"Problematic response: {response_decoded_AE}")
                            if retries < max_retries:
                                print(f'Re-inferencing...')
                                time.sleep(60)
                            else:
                                print(f"Failed to get valid LIST of AE Types after {max_retries} attempts. Skipping Response.")
                                batch_responses_AE.append([])  # Append an empty list if failed
                responses_AE.extend(batch_responses_AE)  # Add this batch's responses to the main list
        if self.verbose is True:
            print(f'Extracted Attributes: {responses_AE}') 
        
        prompts_EE = self.generate_batch(mode='EE')  # batched Prompt for Turn 3: EE (Sentence(String)+Entity Schema(List) -> Entity Types (present in Sentence, List))
        responses_EE = []  # Turn 3에서 추출한 (각 sentence별로 감지된) Entity Types from Entity Schemas가 담긴 list들을 담고 있는, list of list 변수

        for i in tqdm(range(0, len(prompts_EE), batch_size), desc="Processing Batches for Turn 3 (EE)"):
            # Divide into Smaller Batch Sizes
            batch_prompt = prompts_EE[i:i+batch_size]

            # Turn 1: EE (Sentence(String)+Entity Schema(List) -> Entity Types (present in Sentence, List))
            if self.openai_api_key is None:
                inputs_EE = self.tokenizer(batch_prompt, return_tensors="pt", padding=True, truncation=True,).to(self.model.device)
                # allocated_memory = torch.cuda.memory_allocated(device=self.model.device)
                # allocated_memory_mb = allocated_memory / (1024 ** 2)  # Convert bytes to MB
                # print(f"GPU RAM Allocation after tokenization / before generation: {allocated_memory_mb:.2f} MB")
                outputs_EE = self.model.generate(inputs_EE['input_ids'], attention_mask=inputs_EE['attention_mask'], max_new_tokens=512, eos_token_id=self.terminators, do_sample=True, temperature=0.8, top_p=0.9,)

                # 요구된 포멧(List)에 맞는 답변이라면 Turn 3: EE에서 사용하기 위한 Entity Type List로 저장하고, 만약 요구된 포멧을 따르지 않는 답변이 생성됐을 경우 생성 재시도
                for _, output_EE in enumerate(outputs_EE):
                    response_EE = output_EE[inputs_EE['input_ids'].shape[-1]:]
                    response_decoded_EE = self.tokenizer.decode(response_EE, skip_special_tokens=True)  # 이상적으론 List 형태의 String임
                    # Retry mechanism
                    retries = 0
                    while retries < max_retries:
                        try:
                            # Attempt to load the response as List
                            response_list_object = [] if response_decoded_EE == 'None' else [item.strip().rstrip('.') for item in response_decoded_EE.split(",")]
                            response_list_object = list(set(response_list_object))  # Duplicate 제거
                            filtered_EE = [item for item in response_list_object if item != '' and item in self.schemas['objects']]  # Filtering
                            responses_EE.append(filtered_EE[:extraction_count_limit])
                            break  # 정상적으로 list화되는 정상답변이 생성된 경우 바로 루프 탈출
                        except (SyntaxError, NameError, TypeError, ValueError) as e:
                            retries += 1
                            print(f"List Eval Error (Retry {retries}/{max_retries}): {e}"); sys.stdout.flush()
                            print(f"Problematic response: {response_decoded_EE}"); sys.stdout.flush()                        
                            if retries < max_retries:  # Retry Inferencing until correct list structure is generated
                                print(f'Re-inferencing...'); sys.stdout.flush()
                                time.sleep(retry_delay)
                                new_outputs_EE = self.model.generate(inputs_EE['input_ids'], attention_mask=inputs_EE['attention_mask'], max_new_tokens=512, eos_token_id=self.terminators, do_sample=True, temperature=0.8, top_p=0.9,)
                                # 새로 생성한 response를 기존 변수에 대입 (덮어쓰기)
                                response_EE = new_outputs_EE[inputs_EE['input_ids'].shape[-1]:]
                                response_decoded_EE = self.tokenizer.decode(response_EE, skip_special_tokens=True)
                            else:
                                print(f"Failed to get valid LIST of EE Types after {max_retries} attempts. Skipping Response."); sys.stdout.flush()
                                responses_EE.append([])  # 지정된 최대 재시도 횟수 넘길 경우 그냥 스킵하고 빈 리스트 저장
            else:  # OPENAI API
                batch_responses_EE = []
                for prompt in batch_prompt:
                    retries = 0
                    while retries < max_retries:
                        try:
                            response = openai.ChatCompletion.create(
                                model=self.openai_model,
                                messages=prompt,
                                max_tokens=512,
                                temperature=0.8,
                                top_p=0.9,
                            )

                            response_decoded_EE = response['choices'][0]['message']['content']
                            # Attempt to load the response as List
                            response_list_object = [] if response_decoded_EE == 'None' else [item.strip().rstrip('.') for item in response_decoded_EE.split(",")]
                            response_list_object = list(set(response_list_object))  # Remove duplicates
                            filtered_EE = [item for item in response_list_object if item != '' and item in self.schemas['objects']]  # Filtering
                            batch_responses_EE.append(filtered_EE[:extraction_count_limit])
                            break  # Exit loop if successful
                        except (openai.error.OpenAIError, SyntaxError, NameError, TypeError, ValueError, AttributeError) as e:
                            retries += 1
                            print(f"OpenAI API Error / List Eval Error (Retry {retries}/{max_retries}): {e}")
                            print(f"Problematic response: {response_decoded_EE}")
                            if retries < max_retries:
                                print(f'Re-inferencing...')
                                time.sleep(60)
                            else:
                                print(f"Failed to get valid LIST of EE Types after {max_retries} attempts. Skipping Response.")
                                batch_responses_EE.append([])  # Append an empty list if failed
                responses_EE.extend(batch_responses_EE)  # Add this batch's responses to the main list
        if self.verbose is True:
            print(f'Extracted Entities: {responses_EE}') 

        # assert len(responses_RE) == len(response_AE) == len(response_EE) == len(self.list_of_sentences)  # 문장 수와 RE,AE,EE 추출 리스트 수 일치 체크 -> generate_batch() 내에서 자동으로 수행하는 방향으로 수정
        prompts_Final = self.generate_batch(mode='FINAL',re_ae_ee=[responses_RE, responses_AE, responses_EE])
        responses = []  # 최종적으로 생성한 list of {doubles & triples}(dictionaries)들이 담기는 list

        # Divide into Smaller Batch Sizes
        for i in tqdm(range(0, len(prompts_Final), batch_size), desc="Processing batches for Final Extraction"):
            batch_prompt = prompts_Final[i:i+batch_size]

            if self.openai_api_key is None:
                inputs = self.tokenizer(batch_prompt, return_tensors="pt", padding=True, truncation=True,).to(self.model.device)
                # allocated_memory = torch.cuda.memory_allocated(device=self.model.device)
                # allocated_memory_mb = allocated_memory / (1024 ** 2)  # Convert bytes to MB
                # print(f"GPU RAM Allocation after tokenization / before generation: {allocated_memory_mb:.2f} MB")
                outputs = self.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=512, eos_token_id=self.terminators, do_sample=True, temperature=0.8, top_p=0.9,)

                for _, output in enumerate(outputs):
                    response = output[inputs['input_ids'].shape[-1]:]
                    response_decoded = self.tokenizer.decode(response, skip_special_tokens=True)
                    response_decoded = response_decoded.replace("'", '"')

                    if response_decoded[:9] == "assistant":
                        response_decoded = parse_assistant(response_decoded)
                    
                    # Retry mechanism for JSON Decoding Error
                    retries = 0
                    while retries < max_retries:
                        try:
                            # Attempt to load the response as JSON or List
                            # if type(eval(response_decoded)) is list:  # LoRA 모델의 경우 리스트만 출력하는 경향 발견
                            #     response_json_KGs = eval(response_decoded)
                            # else: # 프롬프트로 요구한 바와 같이 딕셔너리 출력한 경우

                            response_json_object = json.loads(response_decoded)  # Convert string to JSON object
                            if isinstance(response_json_object, list):
                                response_json_KGs = response_json_object
                            else:
                                response_json_KGs = response_json_object["extracted_KGs"]
                            responses.append(response_json_KGs)
                            break  # 정상적으로 json decoding 되는 정상답변이 생성된 경우 바로 루프 탈출
                        except json.JSONDecodeError as e:
                            retries += 1
                            print(f"JSON Decode Error (Retry {retries}/{max_retries}): {e}"); sys.stdout.flush()
                            print(f"Problematic response: {response_decoded}"); sys.stdout.flush()
                            # Retry Inferencing until correct json structure is generated
                            if retries < max_retries:
                                print(f'Re-inferencing...'); sys.stdout.flush()
                                time.sleep(retry_delay)
                                new_outputs = self.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=512, eos_token_id=self.terminators, do_sample=True, temperature=0.8, top_p=0.9,)
                                # 새로 생성한 response를 기존 변수에 대입 (덮어쓰기)
                                response = new_outputs[inputs['input_ids'].shape[-1]:]
                                response_decoded = self.tokenizer.decode(response, skip_special_tokens=True)
                            else:  # 지정된 최대 재시도 횟수 넘길 경우 그냥 스킵
                                print(f"Failed to get valid JSON after {max_retries} attempts. Skipping response."); sys.stdout.flush()
                                responses.append([])  # 지정된 최대 재시도 횟수 넘길 경우 그냥 스킵하고 빈 리스트 저장

            else:   #OPENAI API
                # Call the OpenAI API for each batch of prompts
                for prompt in batch_prompt:
                    retries = 0
                    while retries < max_retries:
                        try:
                            # Perform GPT-4 Inference
                            response = openai.ChatCompletion.create(
                                model=self.openai_model,
                                messages=prompt,
                                max_tokens=512,
                                temperature=0.8,
                                top_p=0.9
                            )
                            response_text = response['choices'][0]['message']['content']
                            

                            # Try to parse the response as JSON
                            response_json_object = json.loads(response_text)  # Convert string to JSON object
                            if isinstance(response_json_object, list):
                                response_json_KGs = response_json_object
                            else:
                                response_json_KGs = response_json_object["extracted_KGs"]
                            responses.append(response_json_KGs)
                            break  # 정상적으로 json decoding 되는 정상답변이 생성된 경우 바로 루프 탈출

                        except (json.JSONDecodeError, KeyError) as e:
                            retries += 1
                            print(f"JSON Decode Error (Retry {retries}/{max_retries}): {e}")
                            print(f"Problematic response: {response_text}")
                            if retries < max_retries:
                                print(f"Re-inferencing...")
                                time.sleep(60)
                            else:
                                print(f"Failed to get valid JSON after {max_retries} attempts. Skipping response.")
                                responses.append(None)  # or handle it differently if needed

        return responses  # list of list of dicts
