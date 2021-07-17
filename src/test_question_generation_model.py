"""
Author: @meghabyte

Test the question generation model. For each target difficulty value in the prompts included in the input
prompts file, the program returns the mean and std across LMKT-predicted difficulty values for each
of the 30  generation samples. Requires a file containing prompts (student sequence + difficulty value)
for generation, and LMKT student model for evaluation generation outputs, and a trained question generation model.

Example Usage:
python test_question_generation_model.py --generate_model ../models/question_generation/spanish_qg --lmkt_model ../models/lmkt_student/spanish_student_model --prompts ../data/generation_train_data/spanish_prompts.txt
"""

import numpy as np
import torch
import argparse
import os
import transformers 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from collections import defaultdict
import generate_questions

def resize_prompt(prompt):
    split_prompt = prompt.split(" ")
    new_prompt = split_prompt[-799:]
    new_index = new_prompt.index("<QU>")
    new_prompt = new_prompt[new_index:]
    new_line = " ".join(new_prompt)
    new_line = "<BOS> "+new_line
    print("RESIZE: "+str(new_line))
    return new_line


def generate(prompts, model_directory = "" ):
    generate_args_dict = {"--model_type":"gpt2",
            "--model_name_or_path":"",
            "--length": "20", 
            "--stop_token": "<EOS>",
            "--p": "0.99",
            "--num_return_sequences": "30",
            "--repetition_penalty": "1.0", #Increase Penalty for more novel generations
            "--prompt":""}
    generate_args_dict["--model_name_or_path"]=model_directory
    all_generated = []
    for p in prompts:
        p = p.replace("\n","")
        generate_args_dict["--prompt"]=p
        new_generate_args=[]
        for k in generate_args_dict.keys():
            new_generate_args.append(k)
            new_generate_args.append(generate_args_dict[k])
        sequences = generate_questions.main(new_generate_args)
        generated_sentences = []
        for s in sequences:
            gen_sen = s.replace("<G>","<QU>")+"<AN>"
            gen_sen = " ".join(p.split(" ")[:2])+" "+" ".join(gen_sen.split(" ")[1:])
            generated_sentences.append(gen_sen)
        all_generated.append(generated_sentences)
    return all_generated

def eval(prompts, lmkt_directory, generate_directory):
    print("Loading Model .... ")
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained(lmkt_directory)
    yes_token = tokenizer.encode("<Y>")[0]
    lmkt_model = GPT2LMHeadModel.from_pretrained(lmkt_directory)
    lmkt_model = lmkt_model.cuda()
    print("Generating...")
    lines = generate(prompts, model_directory=generate_directory)
    values_dict = defaultdict(list)
    for l in lines:
        for i in range(len(l)):
            prompt = l[i].replace("\n","")
            prompt = prompt.replace("\n","")
            prompt = "<BOS> "+" ".join(prompt.split(" ")[2: ])
            if(len(prompt.split(" ")) > 800):
                prompt = resize_prompt(prompt)
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs.to(device)
            outputs = lmkt_model(**inputs, return_dict=True)
            logits = outputs.logits.detach().cpu()
            last_token = logits[:, -1].cpu()
            last_token_softmax = torch.softmax(last_token, dim=-1).squeeze()

            #Difficulty (Inverse) is the likelihood of a student answering yes
            difficulty = float(last_token_softmax[yes_token].item())
            difficulty = float(int(difficulty*1000)/1000.0)
            values_dict[l[i].split(" ")[1]].append(difficulty)
    print([[li.split("<QU>")[-1].replace("<AN>","") for li in l] for l in lines])
    for key in values_dict.keys():
        print((key, np.mean(values_dict[key]), np.std(values_dict[key])))

parser = argparse.ArgumentParser(description='Process Test Args.')
parser.add_argument('-p', '--prompts')
parser.add_argument('-m', '--lmkt_model')
parser.add_argument('-g', '--generate_model')
args = parser.parse_args()
f=open(args.prompts,"r")
prompts=f.readlines()
f.close()
eval(prompts, lmkt_directory=args.lmkt_model, generate_directory=args.generate_model)
