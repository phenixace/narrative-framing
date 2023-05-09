from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import torch


LABELS = ['ar', 'hi', 'co', 'mo', 'ec']

class ClimateDataset(Dataset):
    def __init__(self, mode, folder, fold=None):
        super().__init__()
        
        self.text = []
        self.ars = []
        self.his = []
        self.mos = []
        self.cos = []
        self.ecs = []

        if mode == 'unlabelled':
            with open(folder+'/unlabelled.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    temp = line.strip('\n').split('\t')
                    self.text.append(temp[0])
                    self.ars.append(False)
                    self.his.append(False)
                    self.cos.append(False)
                    self.mos.append(False)
                    self.ecs.append(False)
        else:
            with open(folder+'/Fold_'+str(fold)+'/'+mode+'.txt', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    temp = line.strip('\n').split('\t')
                    self.text.append(temp[4])
                    self.ars.append(temp[5]=='True')
                    self.his.append(temp[6]=='True')
                    self.cos.append(temp[7]=='True')
                    self.mos.append(temp[8]=='True')
                    self.ecs.append(temp[9]=='True')


    def __getitem__(self, index):
        return self.text[index], self.ars[index], self.his[index], self.cos[index], self.mos[index], self.ecs[index]

    def __len__(self):
        return len(self.text)

    def get_single_label_dataset(self, label):
        return Single_Dataset(self, label)

class Single_Dataset(Dataset):
    def __init__(self, set, label):
        super().__init__()
        self.texts = []
        self.labels = []

        label_index = LABELS.index(label)

        for i in range(0, len(set)):
            self.texts.append(set[i][0])
            self.labels.append(set[i][1+label_index])

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


    def __len__(self):
        return len(self.labels)

    def update(self, labels):

        assert len(self.labels) == len(labels), "Invalid label assignment, Originally {}, but get {}".format(len(self.labels), len(labels))

        self.labels = labels
    
    def balancing_dataset(self):
        positives = []
        negatives = []
        for i in range(0, len(self.labels)):
            if self.labels[i] == True:
                positives.append(i)
            else:
                negatives.append(i)
        
        if len(positives) > len(negatives):
            for i in range(len(negatives), len(positives)):
                choice = np.random.choice(negatives)
                self.texts.append(self.texts[choice])
                self.labels.append(self.labels[choice])
        else:
            for i in range(len(positives), len(negatives)):
                choice = np.random.choice(positives)
                self.texts.append(self.texts[choice])
                self.labels.append(self.labels[choice])


class EmptyDataset(Dataset):
    def __init__(self, corpus, labels):
        super().__init__()
        self.corpus = corpus
        self.labels = labels

    def __getitem__(self, index):
        return self.corpus[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class Collator(object):
    def __init__(self, tokenizer, max_length, label=False, embeded=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label = label
        self.embeded = embeded

    def __call__(self, batch):
        passages = [ex[0] for ex in batch]
        passages = self.tokenizer.batch_encode_plus(
                    passages,
                    max_length=self.max_length if self.max_length > 0 else None,
                    padding='max_length',
                    return_tensors='pt',
                    truncation=True if self.max_length > 0 else False,)

        if self.embeded:    # use a single classifier
            if self.label:
                targets = []
                for ex in batch:
                    temp = []
                    for i in range(1, 6):
                        # one hot
                        if ex[i] == False:
                            temp.append([1, 0])
                        else:
                            temp.append([0, 1])
                    targets.append(temp)

                targets = torch.tensor(targets)
                # print(targets.shape)
                return passages, targets
            else:
                return passages
        else:   # multiple classifiers
            if self.label:
                targets = torch.tensor([int(ex[1]) for ex in batch])               
                return passages, targets
            else:
                return passages


class SentenceDataset(Dataset):
    def __init__(self, folder, fold, label, mode):
        super().__init__()
        file = folder+'/Fold_'+str(fold)+'/'+label.upper()+'/'+mode+'.txt'

        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        self.texts = []
        self.labels = []

        for line in lines:
            temp = line.split('\t')
            # change here 
            self.texts.append(temp[4:-1])
            self.labels.append(temp[-1].strip('\n') == 'True')


    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.labels)

    def balancing_dataset(self):
        positives = []
        negatives = []
        for i in range(0, len(self.labels)):
            if self.labels[i] == True:
                positives.append(i)
            else:
                negatives.append(i)
        
        if len(positives) > len(negatives):
            for i in range(len(negatives), len(positives)):
                choice = np.random.choice(negatives)
                self.texts.append(self.texts[choice])
                self.labels.append(self.labels[choice])
        else:
            for i in range(len(positives), len(negatives)):
                choice = np.random.choice(positives)
                self.texts.append(self.texts[choice])
                self.labels.append(self.labels[choice])

class SentenceCollator(object):
    def __init__(self, tokenizer, max_length, n_passages):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.n_passages = n_passages

    def __call__(self, batch):
        passages = []
        for ex in batch:
            passages += ex[0][:self.n_passages]

        passages = self.tokenizer.batch_encode_plus(
                    passages,
                    max_length=self.max_length if self.max_length > 0 else None,
                    padding='max_length',
                    return_tensors='pt',
                    truncation=True if self.max_length > 0 else False,)

        targets = torch.tensor([int(ex[1]) for ex in batch])               
        return passages, targets
