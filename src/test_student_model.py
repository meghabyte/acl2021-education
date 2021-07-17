"""
Author: @meghabyte

Test an LM-KT student model on a test set of student prompt sequences and responses. 
Requires:
1.  a trained lmkt student model (model_directory arg)
2. prompts of student sequences, end with an <AN> token for each prompt, 
to predict student likelihood of answering the last question correctly (test_lines arg)
3. the ground truth value for each prompt in test_lines (test_keys arg)
4. an output filename to write results to (output_filename arg)

Example Usage:
main(output_filename="french_results",
	model_directory = "../models/lmkt_student/french_student_model",
	test_lines = "../data/lmkt_test_data/french_test_full", 
	test_keys = "../data/lmkt_test_data/french_test_key")
"""


import torch
import transformers 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import os
import mmap
from tqdm import tqdm
import numpy as np

def get_num_lines(file_path):
	fp = open(file_path, "r+")
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines

def readlines(fn):
	lines = []
	with open(fn) as file:
		for line in tqdm(file, total=get_num_lines(fn)):
			lines.append(line)
	return lines

def main(output_filename, model_directory, test_lines, test_keys):
	device = torch.device("cuda")
	#Put Model Name
	tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
	yes_token = tokenizer.encode("<Y>")[0]
	no_token = tokenizer.encode("<N>")[0]
	model = GPT2LMHeadModel.from_pretrained(model_directory)
	model = model.cuda()
	lines = readlines(test_lines)
	keys = readlines(test_keys)
	print("Running Inputs...")
	next_tokens=[]
	no_tokens = []
	yes_tokens = []
	for li in tqdm(range(len(lines))):
		inputs = tokenizer(lines[li], return_tensors="pt") 
		inputs.to(device)
		outputs = model(**inputs, return_dict=True)
		logits = outputs.logits.detach().cpu()
		last_token = logits[:, -1].cpu()
		last_token_softmax = torch.softmax(last_token, dim=-1).squeeze()
		next_tokens.append(torch.argmax(last_token_softmax))
		no_tokens.append(last_token_softmax[no_token])
		yes_tokens.append(last_token_softmax[yes_token])
	# Log Output File
	output_file = open(output_filename,"w")
	output_file.write("Y tokens: \n")
	output_file.write(str(yes_tokens))
	output_file.write("\n N tokens: \n")
	output_file.write(str(no_tokens))
	preds = np.array(tokenizer.decode(next_tokens, clean_up_tokenization_spaces=True).split(" "))
	answer =  [k.strip() for k in keys]
	output_file.write("\n Answer: \n")
	output_file.write(str(answer))
	answer =  np.array(answer)
	output_file.write("\n Accuracy: "+str((answer==preds).sum()/preds.shape[0]))
	no_indices=(answer=="<N>").nonzero()[0].tolist()
	yes_indices=(answer=="<Y>").nonzero()[0].tolist()
	no_preds = (preds=="<N>").nonzero()[0].tolist()
	yes_preds = (preds=="<Y>").nonzero()[0].tolist()
	output_file.write("\n Y Acc.: "+str((answer==preds)[yes_indices].sum()/len(yes_indices)))
	output_file.write("\n N Acc.: "+str((answer==preds)[no_indices].sum()/len(no_indices)))
	output_file.write("\n Precision:")
	output_file.write(str((answer==preds)[yes_indices].sum()/len(yes_preds)))
	output_file.write("\n Recall:")
	output_file.write(str((answer==preds)[yes_indices].sum()/len(yes_indices)))
	output_file.close()


main(output_filename="french_results",
	model_directory = "../models/lmkt_student/french_student_model",
	test_lines = "../data/lmkt_test_data/french_test_full", 
	test_keys = "../data/lmkt_test_data/french_test_key")