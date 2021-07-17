"""
Author: @meghabyte

Helper functions to process raw Duolingo data files. Extracts per-user data for all tasks with prompts 
(e.g. Reverse Translation), as these contain native-language prompts that the student tries 
to translate in the language they are learning. 

Creates pickle files with per-user data for train, dev, test splits. Process for dev / test
requires joining with a seperate answer key file for student response information. Raw Duolingo
files (e.g. "fr_en.slam.20190204.train") should be downloaded from  https://sharedtask.duolingo.com/2018.html. 


Example Usage:
create_test(dev_file = "fr_en.slam.20190204.test", dev_key_file="fr_en.slam.20190204.test.key")
create_valid(dev_file = "fr_en.slam.20190204.dev", dev_key_file="fr_en.slam.20190204.dev.key")
create_train(train_file="fr_en.slam.20190204.train")
"""

import pandas as pd
import numpy as np 
import pickle
from collections import defaultdict
import mmap
from tqdm import tqdm

def get_num_lines(file_path):
	fp = open(file_path, "r+")
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines


def create_train(train_file, output_fn):
	all_data = defaultdict(list)
	trigger = 0
	prompt = ""
	user = ""
	day = ""
	answer = []
	question = []
	with open(train_file) as file:
		for line in tqdm(file, total=get_num_lines(train_file)):
			if("# prompt" in line): #Task involves English question prompt
				trigger = 1
				prompt = line.replace("# prompt:","").lower().replace("\n","")
				continue
			if("# user" in line and trigger == 1):
				if("reverse_translate" not in line):
					prompt=""
					trigger = 0
					continue
				user = line.split()[1].replace("user:","")
				day = line.split()[3].replace("days:","")
				continue
			if(trigger == 1 and line=="\n"):
				all_data[user].append((prompt, day, answer, question))
				trigger = 0
				prompt = ""
				user = ""
				answer = []
				question = []
				continue
			if(trigger == 1):
				answer.append(int(line.split()[-1]))
				question.append(line.split()[0])
				continue
	pickle.dump(all_data, open(output_fn,"wb"))

def read_key(dev_key_file):
	key_dict = defaultdict()
	file = open(dev_key_file, "r")
	for line in file.readlines():
		line_tup = line.split()
		key_dict[line_tup[0]] = int(line_tup[1])
	return key_dict

def create_valid(dev_file, dev_key_file, output_fn):
	key_dict = read_key(dev_key_file)
	all_data = defaultdict(list)
	valid_data_file = open(dev_file, "r")
	trigger = 0
	prompt = ""
	user = ""
	day = ""
	answer = []
	question = []
	count = 0
	with open(dev_file) as file:
		for line in tqdm(file, total=get_num_lines(dev_file)):
			if("# prompt" in line): #Task involves English question prompt
				trigger = 1
				prompt = line.replace("# prompt:","").lower().replace("\n","")
				continue
			if("# user" in line and trigger == 1):
				user = line.split()[1].replace("user:","")
				day = line.split()[3].replace("days:","")
				continue
			if(trigger == 1 and line=="\n"):
				all_data[user].append((prompt, day, answer, question))
				count += 1
				trigger = 0
				prompt = ""
				user = ""
				answer = []
				question = []
				continue
			if(trigger == 1):
				answer.append(key_dict[line.split()[0]])
				question.append(line.split()[0])
				continue
	pickle.dump(all_data, open(output_fn,"wb"))
