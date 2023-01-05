import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from numpy import argmax
import torch.nn as nn
import torch.nn.functional as F

class Evaluator(object):
    def __init__(self, name = '', classifier='single', mode='macro', detail=False, record_id=False, tmode='dev'):
        self.name = name
        self.classifier = classifier
        self.mode = mode
        self.detail = detail
        self.record_id = record_id
        self.tmode = tmode

    def eval(self, input_dict):
        if self.classifier == 'single':
            if self.tmode == 'test':
                with open('./ckp/' + self.name + '.txt', 'w+', encoding='utf-8') as f:
                    f.write('Pred:\n')
                    f.write(str(input_dict['y_pred']))
                    f.write('\n')
                    f.write('Truth:\n')
                    f.write(str(input_dict['y_true']))

            pred = input_dict['y_pred'].view(-1, 2)
            truth = input_dict['y_true'].view(-1, 2)

            if self.mode == 'macro':

                TP = [0]*5
                FP = [0]*5
                FN = [0]*5
                FP_id = []
                FN_id = []

                ACC = [0]*5

                for i in range(0, len(pred)):
                    if argmax(pred[i]) == argmax(truth[i]) and argmax(truth[i]) == 1:
                        TP[i%5] += 1
                        ACC[i%5] += 1
                    elif argmax(pred[i]) == 0 and argmax(truth[i]) == 1:
                        FN[i%5] += 1
                        FN_id.append(i)
                    elif argmax(pred[i]) == 1 and argmax(truth[i]) == 0:
                        FP[i%5] += 1
                        FP_id.append(i)
                    else:
                        ACC[i%5] += 1
                
                P = [0]*5
                R = [0]*5
                F1 = [0]*5

                for i in range(0, 5):
                    if TP[i] == 0:
                        P[i] = 0
                        R[i] = 0
                        F1[i] = 0
                    else:
                        P[i] = TP[i] / (TP[i] + FP[i])
                        R[i] = TP[i] / (TP[i] + FN[i])
                        F1[i] = 2 * P[i] * R[i] / (P[i] + R[i])
                    
                if self.record_id:
                    with open('./inference.txt', 'w+', encoding='utf-8') as f:
                        f.write('FP:\n')
                        for item in FP_id:
                            temp = F.softmax(pred[item])
                            f.write(str(item)+' '+str(temp[0])+' '+str(temp[1])+'\n')
                        f.write('FN:\n')
                        for item in FN_id:
                            temp = F.softmax(pred[item])
                            f.write(str(item)+' '+str(temp[0])+' '+str(temp[1])+'\n')
                
                if self.detail:
                    return {'Precision':P, 'Recall':R, 'F1':F1, 'Acc': [item*5/len(pred) for item in ACC]}
                else:
                    return {'Precision':sum(P)/5, 'Recall':sum(R)/5, 'F1':sum(F1)/5, 'Acc': sum(ACC)/len(pred)}

            elif self.mode == 'micro':
                TP = 0
                FP = 0
                FN = 0

                for i in range(0, len(pred)):
                    if argmax(pred[i]) == argmax(truth[i]) and argmax(truth[i]) == 1:
                        TP += 1
                    elif argmax(pred[i]) == 0 and argmax(truth[i]) == 1:
                        FN += 1
                    elif argmax(pred[i]) == 1 and argmax(truth[i]) == 0:
                        FP += 1
                if TP == 0:
                    P = 0
                    R = 0
                    F1 = 0
                else:
                    P = TP / (TP + FP)
                    R = TP / (TP + FN)
                    F1 = 2 * P * R / (P + R)

                return {'Precision':P, 'Recall':R, 'F1':F1}

        elif self.classifier == 'naive':
            pred = input_dict['y_pred'] # len*5
            truth = input_dict['y_true'] # len*5

            if self.tmode == 'test':
                with open('./ckp/' + self.name + '.txt', 'w+', encoding='utf-8') as f:
                    f.write('Pred:\n')
                    f.write(str(input_dict['y_pred']))
                    f.write('\n')
                    f.write('Truth:\n')
                    f.write(str(input_dict['y_true']))
            
            if self.mode == 'macro':
                TP = [0]*5
                FP = [0]*5
                FN = [0]*5
                ACC = [0]*5

                for i in range(0, len(pred)):
                    for j in range(0,5):
                        if pred[i][j] == truth[i][j] and truth[i][j] == 1:
                            TP[j] += 1
                            ACC[j] += 1
                        elif pred[i][j] == 0 and truth[i][j] == 1:
                            FN[j] += 1
                        elif pred[i][j] == 1 and truth[i][j] == 0:
                            FP[j] += 1
                        else:
                            ACC[j] += 1

                P = [0]*5
                R = [0]*5
                F1 = [0]*5
                for i in range(0, 5):
                    if TP[i] == 0:
                        P[i] = 0
                        R[i] = 0
                        F1[i] = 0
                    else:
                        P[i] = TP[i] / (TP[i] + FP[i])
                        R[i] = TP[i] / (TP[i] + FN[i])
                        F1[i] = 2 * P[i] * R[i] / (P[i] + R[i])

                if self.detail:
                    return {'Precision':P, 'Recall':R, 'F1':F1, 'Acc':[item/len(pred) for item in ACC]}
                else:
                    return {'Precision':sum(P)/5, 'Recall':sum(R)/5, 'F1':sum(F1)/5, 'Acc':sum(ACC)/(len(pred)*5)}
            elif self.mode == 'micro':
                TP = 0
                FP = 0
                FN = 0
                ACC = 0

                for i in range(0, len(pred)):
                    for j in range(0, 5):
                        if pred[i][j] == truth[i][j] and truth[i][j] == 1:
                            TP += 1
                            ACC += 1
                        elif pred[i][j] == 0 and truth[i][j] == 1:
                            FN += 1
                        elif pred[i][j] == 1 and truth[i][j] == 0:
                            FP += 1
                        else:
                            ACC += 1
                if TP == 0:
                    P = 0
                    R = 0
                    F1 = 0
                else:
                    P = TP / (TP + FP)
                    R = TP / (TP + FN)
                    F1 = 2 * P * R / (P + R)

                return {'Precision':P, 'Recall':R, 'F1':F1, 'Acc': ACC/len(truth)}
        elif self.classifier == 'multiple':
            pred = [argmax(item) for item in input_dict['y_pred']]
            if self.tmode == 'test':
                with open('./ckp/' + self.name + '.txt', 'w+', encoding='utf-8') as f:
                    f.write('Pred:\n')
                    f.write(str(pred))
                    f.write('\n')
                    f.write('Truth:\n')
                    f.write(str(input_dict['y_true']))
            # print(pred)
            truth = input_dict['y_true']
            TP = 0
            FP = 0
            FN = 0
            ACC = 0
            FP_id = []
            FN_id = []

            for i in range(0, len(pred)):
                if pred[i] == truth[i] and truth[i] == 1:
                    TP += 1
                    ACC += 1
                elif pred[i] == 0 and truth[i] == 1:
                    FN += 1
                    FN_id.append(i)
                elif pred[i] == 1 and truth[i] == 0:
                    FP += 1
                    FP_id.append(i)
                else:
                    ACC += 1
            
            if self.record_id:
                with open('./inference.txt', 'w+', encoding='utf-8') as f:
                    f.write('FP:\n')
                    for item in FP_id:
                        temp = F.softmax(input_dict['y_pred'][item])
                        f.write(str(item)+' '+str(input_dict['y_pred'][item][0])+' '+str(input_dict['y_pred'][item][1])+' '+str(temp[0])+' '+str(temp[1])+'\n')
                    f.write('FN:\n')
                    for item in FN_id:
                        temp = F.softmax(input_dict['y_pred'][item])
                        f.write(str(item)+' '+str(input_dict['y_pred'][item][0])+' '+str(input_dict['y_pred'][item][1])+' '+str(temp[0])+' '+str(temp[1])+'\n')

            if TP == 0:
                P = 0
                R = 0
                F1 = 0
            else:
                P = TP / (TP + FP)
                R = TP / (TP + FN)
                F1 = 2 * P * R / (P + R)

            return {'Precision':P, 'Recall':R, 'F1':F1, 'Acc': ACC/len(truth)}
        else:
            raise RuntimeError('Classifer type not defined!')

def filter(text, remove_stopwords=False):
    if remove_stopwords:
        pattern = re.compile("[^ ^a-z^A-Z]")
    else:
        pattern = re.compile("[^.^!^?^'^ ^a-z^A-Z^0-9]")
    text = pattern.sub('', text)

    text = re.sub(" +", " ", text)
    text = re.sub("''+", "", text)
    return text

def stem_and_remove_stopwords(text, lemmatization=False):

    word_tokens = word_tokenize(text)

    if lemmatization:
        lemmatizer = WordNetLemmatizer()

        stop_words = set([lemmatizer.lemmatize(word) for word in stopwords.words('english')])
    else:
        stop_words = set(stopwords.words('english'))

    with open('./utils/stopwords.txt') as f:
        lines = f.readlines()
        for line in lines:
            if lemmatization:
                stop_words.add(lemmatizer.lemmatize(line.strip('\n').strip()))
            else:
                stop_words.add(line.strip('\n').strip())

    filtered_sentence = []

    for w in word_tokens:
        if lemmatization:
            w_processed = lemmatizer.lemmatize(w.strip().lower())
        else:
            w_processed = w.strip().lower()
        if w_processed not in stop_words:
            filtered_sentence.append(w_processed)

    return ' '.join(filtered_sentence)

class Cleaner(object):
    def __init__(self, remove_stopwords=False, lemmatization=False):
        self.remove_stopwords = remove_stopwords
        self.lemmatization = lemmatization
    
    def clean(self, text):
        text = re.sub(r'-', ' ', text)
        text = re.sub(r"$NEWLINE$", " ", text)
        text = re.sub(r"NEWLINE", " ", text)

        # remove urls
        text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', ' ', text)
        # remove @somebody
        text = re.sub(r"@\S+", "", text)

        # remove #topic
        text = re.sub(r"#\S+", "", text)

        # clean unrecognizable characters
        text = filter(text, self.remove_stopwords)

        # text = text.lower()

        if self.remove_stopwords:
            text = stem_and_remove_stopwords(text, self.lemmatization)
        else:
            text = re.sub(" +", " ", text)

        return text


if __name__ == "__main__":
    import torch
    '''import torch.nn as nn

    loss = nn.MSELoss()

    pred = torch.randn(1,5,2)

    print(pred)

    target = torch.tensor([1, 0, 1, 0, 1])

    def myloss(pred, target):
        output = torch.tensor(0.0 ,requires_grad=True)
        for item in pred:
            for i in range(0, len(item)):
                print(item[i], target[i])
                output = output + loss(item[i], target[i])

        return output

    floss = myloss(pred, target)

    floss.backward()

    print(floss)'''

    a = torch.randn(20,5,2)
    b = torch.randn(20,5,2)

    ev = Evaluator()

    print(ev.eval({'y_pred':a, 'y_true':b}))