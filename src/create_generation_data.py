"""
Author: @meghabyte

Create training data for the question generation model. 

This requires:
a file with a list of student prompts (pass as filename), 
an LMKT-model to predict difficulty (pass as model_directory),
a file with a list of questions for the training data  (pass as augmented_fn).
Pass these arguments as a call to the create_train_data() function, which will 
automatically write to and save the training data file.  

"""

import numpy as np
import torch
import os
import transformers 
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def resize_prompt(prompt):
    split_prompt = prompt.split(" ")
    new_prompt = split_prompt[-199:] 
    new_index = new_prompt.index("<QU>")
    new_prompt = new_prompt[new_index:]
    new_line = " ".join(new_prompt)
    new_line = "<BOS> "+new_line
    print("RESIZE: "+str(new_line))
    return new_line
    

def get_lines(filename):
    if(not filename):
        print("No file!")
        return []
    f = open(filename,"r")
    lines = f.readlines()
    print("Done! Read "+str(len(lines))+" lines!")
    f.close()
    return lines


#Given a prompt (student state), and a set of lines to augment it with, return the prompts augmented by each line in a list
def all_prompts(curr_prompt, aug_lines):
    all_prompts = []
    if(len(aug_lines) != 0):
        possible_augs = np.random.default_rng().choice(aug_lines, 5, replace=False)
        for augl in possible_augs:
            augl = augl.replace("\n","")
            new_prompt = curr_prompt
            last_q = new_prompt.rfind("<QU>")
            new_prompt = new_prompt[:last_q]+"<QU> "+augl+" <AN> \n"
            if(len(new_prompt.split(" ")) > 200):
                new_prompt = resize_prompt(new_prompt)
            all_prompts.append(new_prompt)
        return all_prompts
    else:
        if(len(curr_prompt.split(" ")) > 200):
                curr_prompt = resize_prompt(curr_prompt)
        return [curr_prompt]


def create_train_data(prefix="", filename="", model_directory = "../models/lmkt_student/spanish_student_model", augmented_fn=None):
    print("Loading Model .... ")
    device = torch.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained(model_directory)
    yes_token = tokenizer.encode("<Y>")[0]
    model = GPT2LMHeadModel.from_pretrained(model_directory)
    model = model.cuda()
    print("Done!")
    data_f = open(prefix+"_train_"+model_directory.split("_")[-1], "w")
    print("Reading Lines...")
    lines = get_lines(filename)
    augl = get_lines(augmented_fn)
    print("Done! Read "+str(len(lines))+" lines!")
    prop_pos = int(len(lines)/2)
    yes_count = 0
    for i in range(len(lines)):
        if(yes_count >= prop_pos):
            continue
        for prompt in all_prompts(lines[i], augl):
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs.to(device)
            outputs = model(**inputs, return_dict=True)
            logits = outputs.logits.detach().cpu()
            last_token = logits[:, -1].cpu()
            last_token_softmax = torch.softmax(last_token, dim=-1).squeeze()
            likelihood = float(last_token_softmax[yes_token].item())
            likelihood = float(int(likelihood*1000)/1000.0)
            if(likelihood > 0.5):
                yes_count +=1 
            prompt = prompt.replace("<BOS> ","<BOS> "+str(likelihood)+" ").replace("<AN> \n","<EOS>") #changed from new line to prompt
            last_q = prompt.rfind("<QU>")
            prompt = prompt[:last_q]+"<G>"+prompt[last_q+4:]+"\n" 
            data_f.write(prompt)
    data_f.close()

