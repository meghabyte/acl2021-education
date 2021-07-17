"""
Author: @meghabyte

Script to perform ppl ablations on the question generation test sets. 
Use to replicate results in paper on our test splits. 

Example Usage:
Example French Permute: python generation_ppl.py --setting french_test2_rand1 --test_file ../data/generation_test_data/french/rand1_question_test2_trueconf --model_directory ../models/question_generation/french_qg
Example French True: python generation_ppl.py --setting french_test3_true --test_file ../data/generation_test_data/french/question_test3_trueconf_011525192388 --model_directory ../models/question_generation/french_qg
Example Spanish Permute:  python generation_ppl.py --setting spanish_test3_rand3 --test_file  ../data/generation_test_data/spanish/rand3_question_test3_trueconf_025442368075 --model_directory  --model ../models/question_generation/spanish_qg
Example Spanish True: python generation_ppl.py --setting spanish_test2_true --test_file  ../data/generation_test_data/spanish/question_test2_trueconf --model ../models/question_generation/spanish_qg
"""

from LMKT_classes import LineByLineTextDataset, QgGPT2LMHeadModel, QgDataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, Trainer, TrainingArguments
import tempfile
import os
import argparse
import torch
import math
import numpy as np
from datetime import datetime

def main(setting="",test_file="", model_directory=""):
    lines = []
    with open(test_file) as f:
        for line in f:
            lines.append(line.rstrip("\n"))
    range_val = int(len(lines)/5) #5 splits
    perplexities = []
    split = 0
    for itr in np.arange(0, len(lines), range_val):
        split += 1
        print("SPLIT: "+str(split))
        curr_lines = lines[itr:itr+range_val]
        fobj = tempfile.NamedTemporaryFile(mode = "w")
        for cl in curr_lines:
            fobj.write(f"{cl}\n")
        # Set Up Model
        tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
        model = QgGPT2LMHeadModel.from_pretrained(model_directory)
        special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>', 'sep_token': '<SEP>', 'additional_special_tokens': ['<Y>', '<N>', '<QU>', '<AN>', '<G>']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer))
        block_size = tokenizer.max_len
        eval_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=fobj.name, block_size=block_size)
        training_args = TrainingArguments(output_dir="tmp/"+setting+str(itr),overwrite_output_dir=False, do_train=False, do_eval=True, per_device_eval_batch_size=2)
        data_collator = QgDataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, mlm_probability=0.15)
        evaluator = Trainer(model=model, args=training_args, data_collator=data_collator, eval_dataset=eval_dataset, prediction_loss_only=True)
        eval_output = evaluator.evaluate()
        perplexity = math.exp(eval_output["eval_loss"])
        perplexities.append(perplexity)
        print("Perplexity: "+str(perplexity))
    print(perplexities)


parser = argparse.ArgumentParser(description='Process some strings.')
parser.add_argument('-s', '--setting')
parser.add_argument('-t', '--test_file')
parser.add_argument('-m', '--model_directory')
args = parser.parse_args()
main(setting=args.setting, test_file=args.test_file, model_directory=args.model_directory)


