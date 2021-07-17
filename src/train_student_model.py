"""
Author: @meghabyte

Train an LM-KT student model on a file where each line is a student state sequence of <QU> text <AN> (<Y> or <N>) tokens.  

Example Usage:
main(train_data = "../data/lmkt_train_data/spanish_train_full")
"""

import run_language_modeling_student
import os
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
			"--num_train_epochs":"1",
			"--overwrite_output_dir":"True"}

def main(train_data):
	#create arguments
	train_data_keyword = train_data.split("/")[-1]
	output_directory = "model_"+train_data_keyword+"_"+str(datetime.now().time()).replace(".","").replace(":","")
	train_args_dict["--output_dir"]=output_directory
	train_args_dict["--train_data_file"] = train_data
	train_args_dict["--save_total_limit"] = "1"
	train_args_dict["--save_steps"] = "10000"
	#set up train args
	new_train_args = []
	for k in train_args_dict.keys():
		new_train_args.append(k)
		if(train_args_dict[k] != "True"):
			new_train_args.append(train_args_dict[k])

	print("Running Language Modeling for Student Model!")
	run_language_modeling_student.main(new_train_args)

main(train_data = "../data/lmkt_train_data/spanish_train_full")

