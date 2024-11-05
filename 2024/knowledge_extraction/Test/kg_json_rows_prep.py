import json
import os
import re  # regex(regular expression) extraction
from pprint import pprint
from tqdm import tqdm


class KGFilter:
    def __init__(self, kg_json_rows_path, gold_path="dramaQA_KG_GOLD.txt", schema_path="dramaQA_KG_SCHEMAS.txt"):
        self.file_path = kg_json_rows_path
        self.json_list = self.file2data()
        self.gold_path = gold_path
        self.schema_path = schema_path

    def file2data(self):
        '''JSON.ROWS 파일로부터 추출한 Json들을 담은 리스트 반환'''
        with open(self.file_path, 'r') as file:
            json_list  = [json.loads(line) for line in file]
        return json_list
    
    def data2file(self, gold=None, schema=None, cut_idx=None):
        '''extract 함수로 추출/처리 완료하여 반환받은 schemas, extracted_jsons를 각각 파일로 저장'''
        if gold is None and schema is None:
            gold, schema = self.extract()
        
        # 데이터셋 자체의 문제가 있는 경우 특정 인덱스까지만 저장
        if cut_idx is not None:
            gold = gold[:cut_idx-1]
        
        with open(self.gold_path, 'w') as gold_file:
            json.dump(gold, gold_file, indent=4)
        with open(self.schema_path, 'w') as schema_file:
            json.dump(schema, schema_file, indent=4)
        print('file saved')

    
    def extract(self):
        '''Json들을 담은 리스트를 입력받고, Sentence(str) && Schema(dict{objects, attributes, relations}) && Graph(dict{object, attribute} 또는 dict{object, relation, object}) 를 추출하여 dictionary 반환'''
        extracted_jsons = []  # List to store all JSONs
        schemas = {
            'objects': set(),  # Use set to avoid duplicates
            'attributes': set(),
            'relations': set(),
        }

        for json_obj in tqdm(self.json_list, desc="Processing JSON objects"):
            if json_obj['scene_graph'][:1] != "(":
                json_obj['scene_graph'] = "(" + json_obj['scene_graph']
            # Extract all tuples from the scene_graph
            all_tuples = re.findall(r'\(([^)]+)\)', json_obj['scene_graph'])

            extracted_objects = []
            extracted_doubles = []
            extracted_triples = []
            extracted_both = []

            # Process the first tuple for objects
            if all_tuples:
                first_tuple_content = all_tuples[0]
                objects_str = first_tuple_content.strip()
                extracted_objects = [obj.strip() for obj in objects_str.split(',')]
                if len(extracted_objects) != 1:
                    for obj in extracted_objects:
                        if obj != "":
                            schemas['objects'].add(obj)
                else:  # 간혹 objects 튜플 없이 바로 double, triple 나오는 경우도 있으므로 이에 대한 예외처리 필요
                    all_tuples.append(first_tuple_content)  # 뒤에 다시 넣어주기 (생략 방지)


            # Process remaining tuples
            for tuple_content in all_tuples[1:]:  # Skip the first tuple since it's already handled
                parts = [part.strip() for part in tuple_content.split('-')]
                if len(parts) == 2:  # This is a double (object - attribute)
                    obj1, attr = parts
                    extracted_doubles.append({"object": obj1, "attribute": attr})
                    schemas['attributes'].add(attr)
                    if ',' not in obj1:
                        if obj1 != "":
                            schemas['objects'].add(obj1)
                    else:
                        obj1s = [obj.strip() for obj in obj1.split(',')]
                        for obj in obj1s:
                            if obj != "":
                                schemas['objects'].add(obj)
                elif len(parts) == 3:  # This is a triple (object1 - relation - object2)
                    obj1, rel, obj2 = parts
                    extracted_triples.append({"object1": obj1, "relation": rel, "object2": obj2})
                    schemas['relations'].add(rel)
                    if ',' not in obj1:
                        if obj1 != "":
                            schemas['objects'].add(obj1)
                    else:
                        obj1s = [obj.strip() for obj in obj1.split(',')]
                        for obj in obj1s:
                            if obj != "":
                                schemas['objects'].add(obj)
                    if ',' not in obj2:
                        if obj2 != "":
                            schemas['objects'].add(obj2)
                    else:
                        obj2s = [obj.strip() for obj in obj2.split(',')]
                        for obj in obj2s:
                            if obj != "":
                                schemas['objects'].add(obj)

            # Combine doubles and triples
            extracted_both.extend(extracted_doubles)
            extracted_both.extend(extracted_triples)

            extracted_jsons.append({
                "sentence": json_obj['shot_description'],
                "objects": extracted_objects,
                "doubles": extracted_doubles,
                "triples": extracted_triples,
                "gold": extracted_both,
            })

        # Verify length of extracted data
        assert len(self.json_list) == len(extracted_jsons)
        
        # Convert sets to lists for JSON serialization
        schemas_list = {
            'objects': list(schemas['objects']),
            'attributes': list(schemas['attributes']),
            'relations': list(schemas['relations']),
        }
        
        return extracted_jsons, schemas_list

    def extract_old(self):
        '''@@@DEPRECATED due to errors@@@'''
        '''Json들을 담은 리스트를 입력받고, Sentence(str) && Schema(dict{objects, attributes, relations}) && Graph(dict{object, attribute} 또는 dict{object, relation, object}) 를 추출하여 dictionary 반환'''
        extracted_jsons = []  # 모든 Json들을 저장할 리스트 (최종 반환되는 리스트)
        schemas = {
            'objects': set(),  # 중복 방지 위해 set 자료형 사용
            'attributes': set(),  # 중복 방지 위해 set 자료형 사용
            'relations': set(),  # 중복 방지 위해 set 자료형 사용
        }

        for json_obj in tqdm(self.json_list, desc="Processing JSON objects"):
            # 첫 번째 Tuple에서 objects 추출
            first_tuple_content = re.search(r'\(([^)]+)\)', json_obj['scene_graph'])  # Str
            if first_tuple_content:  # 추출된 KG가 비어있는 경우도 있으므로 예외척리
                objects_str = first_tuple_content.group(1).strip()
                extracted_objects = [obj.strip() for obj in objects_str.split(',')]
            else:
                extracted_objects = []

            # 나머지 Tuples에서 추출
            remaining_tuples = re.findall(r'\(([^)]+)\s*-\s*([^)]+)(?:\s*-\s*([^)]+))?\)', json_obj['scene_graph'])

            # 추출된 doubles(Obj-Attr), triples(Obj-Rel-Obj) 저장
            extracted_doubles = []  # double만
            extracted_triples = []  # Triple만
            extracted_both = []  # 둘 다
            
            for tuple_content in remaining_tuples:  # sentence에 대해 추출된 N개의 double, triple들 전부 저장
                obj1 = tuple_content[0].strip()
                schemas['objects'].add(obj1)  # 스키마에 Object 추가 (중복은 자동으로 add함수에서 핸들링됨)
                if tuple_content[2]:  # Obj-Rel-Obj인 경우
                    rel = tuple_content[1].strip()
                    obj2 = tuple_content[2].strip()
                    extracted_triples.append({"object1":obj1, "relation":rel, "object2":obj2})
                    schemas['relations'].add(rel)  # 스키마에 Relation 추가 (중복은 자동으로 add함수에서 핸들링됨)
                    schemas['objects'].add(obj2)  # 스키마에 Object 추가 (중복은 자동으로 add함수에서 핸들링됨)
                else: # Obj-Attr인 경우
                    attr = tuple_content[1].strip()
                    extracted_doubles.append({"object":obj1, "attribute":attr})
                    schemas['attributes'].add(attr)  # 스키마에 Attribute 추가 (중복은 자동으로 add함수에서 핸들링됨)
            # extracted_both는 double, triple 구분 없이 전부 포함하도록 구성
            for double in extracted_doubles:
                extracted_both.append(double)
            for triple in extracted_triples:
                extracted_both.append(triple)

            extracted_jsons.append({
                "sentence": json_obj['shot_description'],
                "objects": extracted_objects,
                "doubles": extracted_doubles,
                "triples": extracted_triples,
                "gold": extracted_both,
            })

        # 추출 길이 체크
        assert len(self.json_list) == len(extracted_jsons)
        
        # 추후 json dump 시 set 자료형 사용불가하므로 serializable한 list 자료형으로 변환
        schemas_list = {
            'objects': list(schemas['objects']),
            'attributes': list(schemas['attributes']),
            'relations': list(schemas['relations']),
        }
        return extracted_jsons, schemas_list
        
            


    
instance_test = KGFilter(
    kg_json_rows_path='../Data/DramaQA_KG/AnotherMissOh_integrated_test_shot.json.rows',
    gold_path='../Data/DramaQA_KG_Processed/KG_GOLD_TEST.json',
    schema_path='../Data/DramaQA_KG_Processed/KG_SCHEMA_TEST.json',
)
gold_test, schemas_test = instance_test.extract()
instance_test.data2file(gold_test, schemas_test, cut_idx=1821)


instance_val = KGFilter(
    kg_json_rows_path='../Data/DramaQA_KG/AnotherMissOh_integrated_val_shot.json.rows',
    gold_path='../Data/DramaQA_KG_Processed/KG_GOLD_VAL.json',
    schema_path='../Data/DramaQA_KG_Processed/KG_SCHEMA_VAL.json',
)
gold_val, schemas_val = instance_val.extract()
instance_val.data2file(gold_val, schemas_val)


instance_train = KGFilter(
    kg_json_rows_path='../Data/DramaQA_KG/AnotherMissOh_integrated_train_shot.json.rows',
    gold_path='../Data/DramaQA_KG_Processed/KG_GOLD_TRAIN.json',
    schema_path='../Data/DramaQA_KG_Processed/KG_SCHEMA_TRAIN.json',
)
gold_train, schemas_train = instance_train.extract()
instance_train.data2file(gold_train, schemas_train)