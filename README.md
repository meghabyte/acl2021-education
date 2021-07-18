# acl2021-education
This repository contains the code for the ACL 2021 paper "Question Generation for Adaptive Education". 

Code for training and testing models is in the `./src` subdirectory. 
Code for replicating reported results and analysis, as well as additional analysis (described below), is in the `./results` subdirectory.
We also provide two scripts to easily play with a LMKT-student knowledge tracing model (`play_student_model.py`) and a question generation model (`play_qg_model.py`). 

If you find this respository useful, please cite:

```
@InProceedings{acl21srivastava,
  title = 	 {Question Generation for Adaptive Education},
  author = 	 {Srivastava, Megha and Goodman, Noah},
  booktitle	=   {Association for Computational Linguistics (ACL)},
  year = 	 {2021},
}
```
The original raw data files are provided by Duolingo and can be accessed at: https://sharedtask.duolingo.com/2018.html. If you use Duolingo data, please make sure to cite:

```
@inproceedings{settles-etal-2018-second,
    title = "Second Language Acquisition Modeling",
    author = "Settles, Burr  and
      Brust, Chris  and
      Gustafson, Erin  and
      Hagiwara, Masato  and
      Madnani, Nitin",
    booktitle = "Proceedings of the Thirteenth Workshop on Innovative Use of {NLP} for Building Educational Applications",
    year = "2018",
}
```

## Dependencies & First Steps

`requirements.txt` contains all required dependencies. To access all data files, run the commands `cd data; unzip data.zip`. To access all models, including those with which you can use the `play_student_model.py` script, run `mkdir models; cd models`, download the file `acl2021_models.zip` (from  https://www.dropbox.com/s/chkdhmn54l2ptzf/acl2021_models.zip?dl=0) into the `models` directory,  and run `unzip acl2021_models.zip`. 

## Play around with LM-KT Student Knowledge Tracing
Once the models have downloaded, you can use interact with trained student models for French and Spanish learners (from English). An example command for French student model is:

``python play_student_model.py -m models/lmkt_student/french_student_model``

To try different prompts (which represents the student question/answer history), modify the `prompts` variable in the script. 

## Play around with Question Generation
You can also interact with trained question generation models for French and Spanish learners. An example command for Spanish learning students is:

``python play_qg_model.py -g models/question_generation/spanish_qg``

To try different prompts (which represents the student question/answer history **and difficulty control**), modify the `prompts` variable in the script. 

## Further Analysis
In this codebase, we include further exploratory analysis that was not included in the main paper. This can be found at `./results/further_analysis`.

In the `lmkt_trends.ipynb` notebook, we investigate what Duolingo questions our LM-KT models predict as easy for all students (e.g. "no, thanks"),  hard for all students (e.g. "why don't you touch the turtle?"), or highly varied in difficulty across students (e.g. "happy new year!"). 

In the `new_vocab.ipynb` notebook, we discover that in addition to generating novel student questions, our question generation model generated **new vocabulary words**, such as **operator** and **teaching**, likely due to words from the same word family appearing in the Duolingo dataset. 

For any questions, please contact megha@cs.stanford.edu! 



 


