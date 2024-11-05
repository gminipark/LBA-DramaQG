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


def parse_assistant(response):
    # Use a regular expression to find content within square brackets
    match = re.search(r'\[(.*?)\]', response)
    if match:
        return match.group(0)  # Return the whole match including brackets
    # return None  # Return None if no match is found
    return ""