"""
Author: @meghabyte

Train a question generation model on a file where each line is a 
consists of a target difficulty (after <BOS>), a student state sequence, a <G> token,
and the generated question .  

Example Usage:
python train_question_generation_model.py --epochs 1 --setting french --train_file ../data/generation_train_data/french_train
"""

import run_language_modeling_question
import os
import argparse
from datetime import datetime
            
train_args_dict = {"--output_dir":"", 
            "--model_type":"gpt2", 
            "--model_name_or_path":"gpt2", 
            "--do_train":"True", 
            "--train_data_file":"", 
            "--eval_data_file":"", 
            "--per_device_train_batch_size":"2",  
            "--per_device_eval_batch_size":"2", 
            "--line_by_line":"True", 
            "--learning_rate":"5e-5", 
            "--num_train_epochs":"2",
            "--overwrite_output_dir":"True"}

def main():
    parser = argparse.ArgumentParser(description='Process some strings.')
    parser.add_argument('-trainf', '--train_file')
    parser.add_argument('-s', '--setting')
    parser.add_argument('-e', '--epochs')
    args = parser.parse_args()
    train_file = args.train_file
    output_directory = args.setting+"_"+str(datetime.now().time()).replace(".","").replace(":","")
    train_args_dict["--output_dir"]=output_directory
    train_args_dict["--train_data_file"] = train_file
    train_args_dict["--num_train_epochs"] = args.epochs
    train_args_dict["--save_total_limit"] = "1"
    train_args_dict["--save_steps"] = "1000"

    #set up train args
    new_train_args = []
    for k in train_args_dict.keys():
        new_train_args.append(k)
        if(train_args_dict[k] != "True"):
            new_train_args.append(train_args_dict[k])
    print(new_train_args)
    run_language_modeling_question.main(new_train_args)

main()  
