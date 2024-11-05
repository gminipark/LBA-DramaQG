import json
import random

def select_random_elements(input_file, output_file, N):
    # Read the input JSON file
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Ensure N does not exceed the number of elements in the input
    if N > len(data):
        raise ValueError(f"N cannot be greater than the number of elements in the input file (found {len(data)} elements).")
    
    # Randomly select N elements
    selected_elements = random.sample(data, N)
    
    # Write the selected elements to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(selected_elements, f, indent=4)


# select_random_elements('./Data/DramaQA_KG_Processed/KG_GOLD_TEST.json', './Data/DramaQA_KG_Processed/KG_GOLD_TEST_500.json', 500)

# with open('./Data/DramaQA_KG_Processed/KG_GOLD_TEST_500.json', 'r') as f:
#     data = json.load(f)
# print(len(data))