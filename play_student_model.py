"""
Author: @meghabyte

Script to interact with an LMKT student model. Returns likelihood of a student answering the
last question in the prompt sequence correctly.
Modify the prompts variable to test specific student state sequences. 
Pass in a trained student model with the --lmkt_model argument.

Example Usage:
python play_student_model.py -m models/lmkt_student/french_student_model
"""

import torch
import transformers 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import mmap
from tqdm import tqdm
import numpy as np
import argparse

prompts = [
"<BOS> <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> we drink. <AN> <Y> <QU> he loves you. <AN>",
"<BOS> <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> we drink. <AN> <Y> <QU> i love you. <AN> <Y> <QU> he loves you. <AN>",
"<BOS> <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> i love you. <AN> <N> <QU> we drink. <AN> <Y> <QU> i love you. <AN> <Y> <QU> he loves you. <AN>",
"<BOS> <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> i love you. <AN> <N> <QU> she loves you. <AN> <N> <QU> we drink. <AN> <Y> <QU> i love you. <AN> <Y> <QU> he loves you. <AN>",
"<BOS> <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> i love you. <AN> <N> <QU> she loves you. <AN> <Y> <QU> we drink. <AN> <Y> <QU> i love you. <AN> <Y> <QU> he loves you. <AN>",
"<BOS> <QU> a cat <AN> <N> <QU> the man <AN> <N> <QU> the boy <AN> <Y> <QU> i eat. <AN> <Y> <QU> i am calm. <AN> <Y> <QU> we eat. <AN> <Y> <QU> i love you. <AN> <N> <QU> he loves you. <AN> <Y> <QU> we drink. <AN> <Y> <QU> i love you. <AN> <Y> <QU> he loves you. <AN>"]

def run_prompt(model_directory):
	device = torch.device("cuda")
	#Put Model Name
	tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
	yes_token = tokenizer.encode("<Y>")[0]
	no_token = tokenizer.encode("<N>")[0]
	model = GPT2LMHeadModel.from_pretrained(model_directory)
	model = model.cuda()
	print("Running Prompts")
	for prompt in prompts:
		print(prompt)
		inputs = tokenizer(prompt, return_tensors="pt") 
		inputs.to(device)
		outputs = model(**inputs, return_dict=True)
		logits = outputs.logits.detach().cpu()
		last_token = logits[:, -1].cpu()
		last_token_softmax = torch.softmax(last_token, dim=-1).squeeze()
		print("Likelihood: "+str(last_token_softmax[yes_token]))
		print("\n")

parser = argparse.ArgumentParser(description='Process Test Args.')
parser.add_argument('-m', '--lmkt_model')
args = parser.parse_args()
run_prompt(model_directory=args.lmkt_model)
