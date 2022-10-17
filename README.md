# GUS
This is the official code and data for paper "A Generative User Simulator with GPT-based Architecture and Goal State
Tracking for Reinforced Multi-Domain Dialog Systems"
## Requirements
After you create an environment with `python 3.6`, the following commands are recommended to install required packages.
* pip install torch==1.5
* pip install transformers==3.5
* pip install spacy==3.1
* python -m spacy download en_core_web_sm
* pip install sklearn
* pip install tensorboard

Besides, you need to install the [standard evaluation repository](https://github.com/Tomiinek/MultiWOZ_Evaluation) for corpus-based evaluation, in which we change the references in `mwzeval/utils.py/load_references()` to 'damd', since we adopt the same delexicalization as [DAMD](https://github.com/thu-spmi/damd-multiwoz). 

[ConvLab-2](https://github.com/thu-coai/Convlab-2) is also needed to interact with the [agenda based user simulator (ABUS)](https://aclanthology.org/N07-2038/) on MultiWOZ.
## Data Preparation
Based on the data preprocessing of [DAMD](https://github.com/thu-spmi/damd-multiwoz), we add goal state span ('gpan') and user act span ('usr_act') for each turn, which is scripted in [prepare_data.py](./prepare_data.py). 
For convenience, we provide preprocessed training data, database files and random generated testing goals in the format of `.zip` file. Execute following commands to unzip them
```
unzip db.zip -d ./db/
cd analysis
unzip goals.zip -d ./
cd ../data/multi-woz-2.1-processed
unzip data.zip -d ./
```
## Supervised Training
To pretrain the dialog system (DS), run
```
bash pretrain_ds.sh $GPU
```
To pretrain the user simulator (US), run
```
bash pretrain_us.sh $GPU
```
## RL Training
To implement RL experiments, run
```
bash run_RL.sh $GPU
```
If necessary, you can change the settings in [run_RL.sh](./run_RL.sh), which is described in [config.py](./config.py).
For instance, 
* Set `interact_with_ABUS=True` in [run_RL.sh](./run_RL.sh) to train the DS with ABUS.
* Set `simple_reward=True` in [run_RL.sh](./run_RL.sh) to take Success as reward.
* Set `full_goal=True` in [run_RL.sh](./run_RL.sh) to replace the goal state with the full goal span in US.
## Evaluation
### Corpus-base Evaluation
To evaluate DSs/USs based on the annotations in corpus, run
```
bash test.sh $GPU $DS_PATH
bash test_us.sh $GPU $US_PATH
```
### Interaction with GUS
To evaluate the interaction quality of DSs and GUS, run
```
bash eval_RL.sh $GPU $DS_PATH $US_PATH
```
### Interaction with ABUS
To test with ABUS, run
```
python test_with_ABUS.py --device $DEVICE --path $DS_PATH
```
Note that you need to save the interaction results and evaluate them with
```
python analyze_result.py --result_path $RESULT_PATH
```
