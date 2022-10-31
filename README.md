# Framing Detection

## Requirements
* torch
* argparse
* numpy
* tqdm
* transformers
* nltk

## Usage
### exploratory analysis
Please check the directory `./exploratory analysis/`
* dataset.py: transform the dataset format
* exploratory_analysis.ipynb: conduct exploratory analysis
* naive_baselines.ipynb: Baseline models
* pre_training: further pre-training PLMs on our unlabelled dataset
* sentence_bert.ipynb: sentence-BERT test and visualization
* topic_modelling.ipynb: conduct LDA topic modelling
* utils.py: auxiliary functions

### run training 
base
```
python training_base.py

Framing Detection in Climate Change

optional arguments:
  -h, --help            show this help message and exit
  --random_seed RANDOM_SEED
                        Random Seed for the program
  --device DEVICE       Selecting running device (default:cuda:0)
  --lr LR               learning rate (default: 2e-5)
  --lm LM               pre-trained language model
  --model MODEL         model structure to use
  --dataset DATASET     entire dataset file path
  --specified_label SPECIFIED_LABEL
                        label for model BERT4SIN
  --fine_tuning         fine tune the weights of bert
  --dataset_balancing   Balance the label distribution in the dataset
  --max_len MAX_LEN     max length the input can take (default: 512)
  --fold FOLD           We do 5-fold validation, select fold number here (default: 1)
  --ckp_path CKP_PATH   further pretrained model path
  --batch_size BATCH_SIZE
                        batch size for training (default: 8)
  --epochs EPOCHS       number of training epochs (default: 20)
  --log_dir LOG_DIR     tensorboard log directory
  --checkpoint_dir CHECKPOINT_DIR
                        directory to save checkpoint
```
sentence-bert
```
python training_stb.py

Framing Detection in Climate Change

optional arguments:
  -h, --help            show this help message and exit
  --random_seed RANDOM_SEED
                        Random Seed for the program
  --device DEVICE       Selecting running device (default:cuda:0)
  --dataset DATASET     dataset folder path
  --lr LR               learning rate (default: 2e-6)
  --lm LM               pre-trained language model
  --max_len MAX_LEN     max length the input can take (default: 64)
  --fold FOLD           We do 5-fold validation, select fold number here (default: 1)
  --n_passages N_PASSAGES
                        How many sentences to select, (1-5)
  --batch_size BATCH_SIZE
                        batch size for training (default: 4)
  --epochs EPOCHS       number of training epochs (default: 20)
  --specified_label SPECIFIED_LABEL
                        label for training
  --fusion FUSION       Fusion Strategy
  --fine_tuning         fine tune the weights of bert
  --dataset_balancing   Balance the label distribution in the dataset
  --log_dir LOG_DIR     tensorboard log directory
  --checkpoint_dir CHECKPOINT_DIR
                        directory to save checkpoint
```
#### run batch training
`Remember to edit the codes first to run batch training`
```
python run_training.py
```