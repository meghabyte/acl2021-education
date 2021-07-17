"""
Author: @meghabyte

Script to interact with a question generation model. Returns a list of generated questions.
Modify the prompts variable to test specific student state / difficulty value sequences. 
Pass in a trained question generation model with the --generate_model argument.

Example Usage:
python play_qg_model.py -g models/question_generation/french_qg
"""


import torch
import transformers 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import mmap
from tqdm import tqdm
import numpy as np
import argparse
import src
from src import generate_questions
from src import LMKT_classes

prompts = ["<BOS> 90.0 <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> i love you. <AN> <N> <QU> she loves you. <AN> <N> <QU> we drink. <AN> <Y> <QU> i love you. <AN> <Y> <G>",
"<BOS> 70.0 <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> i love you. <AN> <N> <QU> she loves you. <AN> <N> <QU> we drink. <AN> <Y> <QU> i love you. <AN> <Y> <G>",
"<BOS> 20.0 <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> i love you. <AN> <N> <QU> she loves you. <AN> <N> <QU> we drink. <AN> <Y> <QU> i love you. <AN> <Y> <G>"]

def run_prompt(model_directory):
    device = torch.device("cuda")
    generate_args_dict = {"--model_type":"gpt2",
            "--model_name_or_path":"",
            "--length": "20", 
            "--stop_token": "<EOS>",
            "--p": "0.99",
            "--num_return_sequences": "5",
            "--repetition_penalty": "1.0", #Increase the penalty value for more novel questions 
            "--prompt":""}
    generate_args_dict["--model_name_or_path"]=model_directory
    all_generated = []
    print("Running Prompts")
    for prompt in prompts:
        print(prompt)
        prompt = prompt.replace("\n","")
        generate_args_dict["--prompt"]=prompt 
        new_generate_args=[]
        for k in generate_args_dict.keys():
            new_generate_args.append(k)
            new_generate_args.append(generate_args_dict[k])
        sequences = generate_questions.main(new_generate_args)
        generated_sentences = []
        for s in sequences:
            generated_sentences.append(s.split("<G>")[-1])
        print("Generated Questions: "+str(generated_sentences))
        print("\n")

parser = argparse.ArgumentParser(description='Process Test Args.')
parser.add_argument('-g', '--generate_model')
args = parser.parse_args()
run_prompt(model_directory=args.generate_model)
