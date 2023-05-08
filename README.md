# Framing Detection
This repo releases the data and codes for ACL 2023 paper "Conflicts, Villains, Resolutions:
Towards models of Narrative Media Framing"

## Requirements
* torch == 1.13.1
* argparse == 1.1
* numpy == 1.21.5
* tqdm == 4.64.1
* transformers == 4.28.0
* nltk == 3.7

Other versions might also work, but the default settings are recommended if you encounter any version inconsistency problems.

## Data
There are two processed dataset in the current repo.
* [`./dataset_original`](./dataset_original)

Structure:
```
- Fold_(1-5)
  - dev.txt
  - test.txt
  - train.txt
```

Format:
```
ID  Source  Bias  Time  Full_News_Content  AR  HI  CO  MO  EC
```
* [`./dataset_sentence`](./dataset_sentence)

Structure:
```
- Fold_(1-5)
  - LABEL
    - dev.txt
    - test.txt
    - train.txt
```

Format:
```
ID  Source  Bias  Time  S1  S2  S3  S4  S5  Remaining_Sentences_RankedByRelateness  Full_News_Content  LABEL
```

## Usage
### exploratory analysis
Please check the directory [`./exploratory_analysis/`](./exploratory_analysis/)
* [dataset.py](./exploratory_analysis/dataset.py): transform the dataset format
* [exploratory_analysis.ipynb](./exploratory_analysis/exploratory_analysis.ipynb): conduct exploratory analysis
* [naive_baselines.ipynb](./exploratory_analysis/naive_baselines.ipynb): Baseline models
* [sentence_bert.ipynb](./exploratory_analysis/sentence_bert.ipynb): sentence-BERT test and visualization
* [topic_modelling.ipynb](./exploratory_analysis/topic_modelling.ipynb): conduct LDA topic modelling
* [unsupervisedRBF.ipynb](./exploratory_analysis/unsupervisedRBF.ipynb): conduct unsupervised RBF
* [utils.py](./exploratory_analysis/utils.py): auxiliary functions
* [annotated_data_500](./exploratory_analysis/annotated_data_500): The original dataset with labels
* [unlabelled_articles_17K](./exploratory_analysis/unlabelled_articles_17K): The original dataset without labels
* [lda_results](./exploratory_analysis/lda_results): LDA Results of the original dataset
* [output](./exploratory_analysis/output): News Distribution Plot
* [utils](./exploratory_analysis/utils): The list of stopwords 


### run training 
#### [base](./training_base.py)
```
python training_base.py

Framing Detection in Climate Change

optional arguments:
  -h, --help            show this help message and exit
  --random_seed RANDOM_SEED
                        Random Seed for the program
  --device DEVICE       Selecting running device (default:cuda:0)
  --lr LR               learning rate (default: 2e-6)
  --lm LM               pre-trained language model
  --model MODEL         model structure to use
  --dataset DATASET     dataset folder path
  --specified_label SPECIFIED_LABEL
                        label for training
  --fine_tuning         fine tune the weights of bert
  --dataset_balancing   Balance the label distribution in the dataset
  --max_len MAX_LEN     max length the input can take (default: 256)
  --fold FOLD           We do 5-fold validation, select fold number here (range: 1~5)
  --ckp_path CKP_PATH   further pretrained model path
  --batch_size BATCH_SIZE
                        batch size for training (default: 16)
  --epochs EPOCHS       number of training epochs (default: 20)
  --log_dir LOG_DIR     tensorboard log directory
  --checkpoint_dir CHECKPOINT_DIR
                        directory to save checkpoint
```
#### [RBF](./training_rbf.py)
```
python training_rbf.py

Framing Detection in Climate Change

optional arguments:
  -h, --help            show this help message and exit
  --random_seed RANDOM_SEED
                        Random Seed for the program
  --device DEVICE       Selecting running device (default:cuda:0)
  --dataset DATASET     dataset folder path
  --lr LR               learning rate (default: 2e-6)
  --lm LM               pre-trained language model
  --max_len MAX_LEN     max length the input can take (default: 256)
  --fold FOLD           We do 5-fold validation, select fold number here (range: 1~5)
  --n_passages N_PASSAGES
                        How many channels to select (range: 1~5), RBF-C <=> n_passages=5, RBF-C -a <=> n_passages=4, RBF-C -a-t <=> n_passages=3
  --batch_size BATCH_SIZE
                        batch size for training (default: 8)
  --epochs EPOCHS       number of training epochs (default: 20)
  --specified_label SPECIFIED_LABEL
                        label for training
  --fusion FUSION       Fusion Strategy
  --fine_tuning         fine tune the weights of PLM
  --dataset_balancing   Balance the label distribution in the dataset
  --log_dir LOG_DIR     tensorboard log directory
  --checkpoint_dir CHECKPOINT_DIR
                        directory to save checkpoint
```
#### run batch training
`Remember to edit the codes first before running batch training`
```
python ./scripts/run_training_x.py
```
`Training on the cloud clusters (Please make sure the environment has been set up.)`
```
sbatch ./scripts/run_training_x.slurm
```
#### Compared to KNN+TF-IDF
Please run the related codes in jupyter notebook [Naive Baselines](./exploratory_analysis/naive_baselines.ipynb)

## Citation
```
@inproceedings{rbf_lea_2023,
  title={Conflicts, Villains, Resolutions: Towards models of Narrative Media Framing},
  author={Lea Frermann and Jiatong Li and Shima Khanehzar and Gosia Mikolajczak},
  booktitle={ACL},
  year={2023}
}
```
