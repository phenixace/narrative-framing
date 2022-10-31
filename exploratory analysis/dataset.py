from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import json

import torch
from utils import Cleaner
from transformers import BertTokenizer

LABELS = ['ar', 'hi', 'mo', 'co', 'ec']

class SupervisedDataset(Dataset):
    def __init__(self, corpus, annotated_ids, annotated_labels, remove_stopwords=False, lemmatization=False):
        super().__init__()
        # get id from labelled corpus
        self.labelled_corpus = []
        self.cleaner = Cleaner(remove_stopwords=remove_stopwords, lemmatization=lemmatization)
        
        for i in range(0, len(annotated_ids)):
            flag = False
            for j in range(0, corpus.shape[0]):
                if annotated_ids[i] == corpus['id'][j]:
                    self.labelled_corpus.append(corpus['content'][j])
                    flag = True
                    break
            if flag == False:
                print('Missing File', annotated_ids[i])

        self.labels = {'ar':set(), 'hi':set(), 'mo':set(), 'co':set(), 'ec':set()}

        annotated_file = open(annotated_labels, 'r', encoding='utf-8')

        annotated_file = json.load(annotated_file)

        keys = list(annotated_file.keys())
        for i in range(0, len(keys)):
            item = keys[i]
            temp = annotated_file[item]
            # Attribution of Responsibility
            # question answer may not exist, like hi1, so here we must examine the key
            if 'ar2' in temp.keys():
                if temp['ar2'][0] == 'yes':
                    self.labels['ar'].add(i)
            if 'ar6' in temp.keys():
                if temp['ar6'][0] == 'yes':
                    self.labels['ar'].add(i)

            # Human Interest
            if 'hi1' in temp.keys():
                if temp['hi1'][0] == 'yes':
                    self.labels['hi'].add(i)
            if 'hi2' in temp.keys():
                if temp['hi2'][0] == 'yes':
                    self.labels['hi'].add(i)
            if 'hi5' in temp.keys():
                if temp['hi5'][0] == 'yes':
                    self.labels['hi'].add(i)

            # Morality
            if 'mo1' in temp.keys():
                if temp['mo1'][0] == 'yes':
                    self.labels['mo'].add(i)
            if 'mo2' in temp.keys():
                if temp['mo2'][0] == 'yes':
                    self.labels['mo'].add(i)

            # Conflict
            if 'co1' in temp.keys():
                if temp['co1'][0] == 'yes':
                    self.labels['co'].add(i)
            if 'co2' in temp.keys():
                if temp['co2'][0] == 'yes':
                    self.labels['co'].add(i)
            if 'co3' in temp.keys():
                if temp['co3'][0] == 'yes':
                    self.labels['co'].add(i)

            # Economic
            if 'ec1' in temp.keys():
                if temp['ec1'][0] == 'yes':
                    self.labels['ec'].add(i)
            if 'ec2' in temp.keys():
                if temp['ec2'][0] == 'yes':
                    self.labels['ec'].add(i)
            if 'ec3' in temp.keys():
                if temp['ec3'][0] == 'yes':
                    self.labels['ec'].add(i)

    def __getitem__(self, index):
        return self.cleaner.clean(self.labelled_corpus[index]), [int(index in self.labels[key]) for key in LABELS]

    def __len__(self):
        return len(self.labelled_corpus)

    def save2text(self, path, label, fold):
        f_train = open(path + '/train.txt', 'w+', encoding='utf-8')
        f_valid = open(path + '/dev.txt', 'w+', encoding='utf-8')
        f_test  = open(path + '/test.txt', 'w+', encoding='utf-8')

        if label:
            for i in range(0, len(self.labelled_corpus)):
                if i%5 == ((0 + fold-1) % 5) or i%5 == ((1 + fold-1) % 5) or i%5 == ((2 + fold-1) % 5):
                    f_train.write(self.cleaner.clean(self.labelled_corpus[i])+'\t'+str(i in self.labels[label])+'\n')
                elif i%5 == ((3 + fold-1) % 5):
                    f_valid.write(self.cleaner.clean(self.labelled_corpus[i])+'\t'+str(i in self.labels[label])+'\n')
                elif i%5 == ((4 + fold-1) % 5):
                    f_test.write(self.cleaner.clean(self.labelled_corpus[i])+'\t'+str(i in self.labels[label])+'\n')
        else:
            for i in range(0, len(self.labelled_corpus)):
                if i%5 == ((0 + fold-1) % 5) or i%5 == ((1 + fold-1) % 5) or i%5 == ((2 + fold-1) % 5):
                    f_train.write(self.cleaner.clean(self.labelled_corpus[i])+'\t'+str(i in self.labels['ar'])+'\t'+str(i in self.labels['hi'])+'\t'+str(i in self.labels['mo'])+'\t'+str(i in self.labels['co'])+'\t'+str(i in self.labels['ec'])+'\n')
                elif i%5 == ((3 + fold-1) % 5):
                    f_valid.write(self.cleaner.clean(self.labelled_corpus[i])+'\t'+str(i in self.labels['ar'])+'\t'+str(i in self.labels['hi'])+'\t'+str(i in self.labels['mo'])+'\t'+str(i in self.labels['co'])+'\t'+str(i in self.labels['ec'])+'\n')
                elif i%5 == ((4 + fold-1) % 5):
                    f_test.write(self.cleaner.clean(self.labelled_corpus[i])+'\t'+str(i in self.labels['ar'])+'\t'+str(i in self.labels['hi'])+'\t'+str(i in self.labels['mo'])+'\t'+str(i in self.labels['co'])+'\t'+str(i in self.labels['ec'])+'\n')
        f_train.close()
        f_valid.close()
        f_test.close()

class UnsupervisedDataset(Dataset):
    def __init__(self, corpus, annotated_ids, remove_stopwords=False, lemmatization=False):
        super().__init__()
        self.cleaner = Cleaner(remove_stopwords=remove_stopwords, lemmatization=lemmatization)
        self.unlabelled_corpus = []

        for i in range(0, corpus.shape[0]):
            flag = False
            for j in range(0, len(annotated_ids)):
                if annotated_ids[j] == corpus['id'][i]:
                    flag = True
                    break

            if flag == False:
                self.unlabelled_corpus.append(corpus['content'][i])

    def __getitem__(self, index):
        return self.cleaner.clean(self.unlabelled_corpus[index])

    def __len__(self):
        return len(self.unlabelled_corpus)

    def save2text(self, path):
        with open(path, 'w+', encoding='utf-8') as f:
            for i in range(0, len(self.unlabelled_corpus)):
                f.write(self.cleaner.clean(self.unlabelled_corpus[i])+'\tFalse\n')


class ClimateDataset(Dataset):
    def __init__(self, filename, annotated_id_file, annotated_file, remove_stopwords=False, lemmatization=False):
        super().__init__()
        self.corpus = pd.read_csv(filename)
        self.cleaner = Cleaner(remove_stopwords=remove_stopwords, lemmatization=lemmatization)
        self.remove_stopwords = remove_stopwords
        self.lemmatization = lemmatization
        f = open(annotated_id_file)
        content = f.read()
        f.close()
        examples = content.split("==========================================")[1:]

        self.annotated_ids = []

        for item in examples:
            temp = item.split('\n')
            self.annotated_ids.append(temp[2][4:].strip())

        self.annotated_file = annotated_file


    def __getitem__(self, index):
        return self.cleaner.clean(self.corpus['content'][index])

    def __len__(self):
        return self.corpus.shape[0]

    def get_labelled_dataset(self):
        return SupervisedDataset(self.corpus, self.annotated_ids, self.annotated_file, self.remove_stopwords, self.lemmatization)

    def get_unlabelled_dataset(self):
        return UnsupervisedDataset(self.corpus, self.annotated_ids, self.remove_stopwords, self.lemmatization)
    
class Collator(object):
    def __init__(self, tokenizer, max_length, label=True, multiple_label=False, specified_label=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label = label
        self.multiple_label = multiple_label
        self.specified_label = specified_label

        self.label_index = {'ar':0, 'hi':1, 'mo':2, 'co':3, 'ec':4}

    def __call__(self, batch):
        passages = [ex[0] for ex in batch]

        if self.label:
            if self.multiple_label:

                targets = {}
                for item in LABELS:
                    targets[item] = torch.tensor([ex[1][self.label_index[item]] for ex in batch])

            elif self.specified_label:
                targets = torch.tensor([ex[1][self.label_index[self.specified_label]] for ex in batch])

            else:
                targets = torch.cat([F.one_hot(torch.tensor(ex[1]), num_classes=2) for ex in batch], dim=-2).view(-1, 5, 2)
        
        
        passages = self.tokenizer.batch_encode_plus(
            passages,
            max_length=self.max_length if self.max_length > 0 else None,
            padding='max_length',
            return_tensors='pt',
            truncation=True if self.max_length > 0 else False,)

        if self.label:
            return passages, targets
        else:
            return passages

if __name__ == "__main__":
    for fold in range(3, 6):
        path = './SPT-NFD/dataset_sentence/Fold_'+str(fold)+'/'

        entire_dataset = ClimateDataset("./unlabelled_articles_17K/opinion_climate_all_with_bias.csv", './annotated_data_500/pretty_0611_lcad.txt', "./annotated_data_500/0611_majority.json")

        labelled_set = entire_dataset.get_labelled_dataset()

        unlablled_set = entire_dataset.get_unlabelled_dataset()

        from sentence_transformers import SentenceTransformer, util
        from nltk.tokenize import sent_tokenize

        model = SentenceTransformer("all-mpnet-base-v2")
        queries = ['attribution of responsibility', 'human interest', 'morality', 'conflict', 'economy']
        labels = ['AR','HI','MO','CO','EC']
        query_embeddings = model.encode(queries)

        for i in range(0, len(labelled_set)):
            sentences = sent_tokenize(labelled_set[i][0])
            embeddings = model.encode(sentences)
            cos_sim = util.cos_sim(query_embeddings, embeddings)

            for j in range(0, len(queries)):
                # print(queries[j], labels[j], LABELS[j])
                sentence_combinations = []
                for k in range(0, len(sentences)):
                    sentence_combinations.append([cos_sim[j][k], j, k])
                sentence_combinations = sorted(sentence_combinations, key=lambda x: x[0], reverse=True)

                cur_path = path + labels[j]
                f_train = open(cur_path + '/train.txt', 'a+', encoding='utf-8')
                f_valid = open(cur_path + '/dev.txt', 'a+', encoding='utf-8')
                f_test  = open(cur_path + '/test.txt', 'a+', encoding='utf-8')


                text = []
                label = str(labelled_set[i][1][j] == 1)

                for score, x, y in sentence_combinations[:5]:
                    text.append(sentences[y])

                while len(text) < 5:
                    text = [text[0]] + text

                text = '\t'.join(text)

                if i%5 == ((0 + fold-1) % 5) or i%5 == ((1 + fold-1) % 5) or i%5 == ((2 + fold-1) % 5):
                    f_train.write(text+'\t'+label+'\n')
                elif i%5 == ((3 + fold-1) % 5):
                    f_valid.write(text+'\t'+label+'\n')
                elif i%5 == ((4 + fold-1) % 5):
                    f_test.write(text+'\t'+label+'\n')

                f_train.close()
                f_test.close()
                f_valid.close()