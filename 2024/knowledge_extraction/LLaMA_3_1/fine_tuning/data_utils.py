import os
import json
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
tqdm.pandas()  # Enable the tqdm progress bar for pandas
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
import random
random.seed(42)


def plot_token_count_distribution(df_col):
    plt.hist(df_col, weights=np.ones(len(df_col)) / len(df_col))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.xlabel("FewShot-Tokens")
    plt.xlabel("Tokens")
    plt.ylabel("Percentages")
    print("Saving Plot for Maximum Token Length Distributions...")
    plt.savefig(f"./Data/DramaQA_KG_Processed/TrainingData/Token_Counts_{df_col.name}.png")
    plt.close()
    print("Done")


def split(df):
    train, temp = train_test_split(df, test_size=0.2)
    val, test = train_test_split(temp, test_size=0.2)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)