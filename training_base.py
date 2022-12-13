import os
import random
import torch
import argparse

from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn

import numpy as np
from tqdm import tqdm

from dataset import ClimateDataset, Collator
from utils import Evaluator
from models import FD_BASE, FD_SIN
from transformers import BertTokenizer, LongformerTokenizer

torch.set_printoptions(threshold=np.inf)

def train_0(model, loader, optimizer, criterion, device):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        for key in batch[0].keys():
            batch[0][key] = batch[0][key].to(device)

        pred = model(batch[0])
        #print(pred, batch[1])
        optimizer.zero_grad()

        loss = criterion(pred, batch[1].to(device).float())
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def train_1(model, loader, optimizer, criterion, device):
    model.train()
    loss_accum = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        for key in batch[0].keys():
            batch[0][key] = batch[0][key].to(device)

        pred = model(batch[0])
        #print(pred, batch[1])
        optimizer.zero_grad()

        loss = criterion(pred, batch[1].to(device))
        loss.backward()
        optimizer.step()

        loss_accum += loss.detach().cpu().item()

    return loss_accum / (step + 1)

def eval(model, loader, evaluator, criterion, device):
    model.eval()
    y_true = []
    y_pred = []

    loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):

        for key in batch[0].keys():
            batch[0][key] = batch[0][key].to(device)

        with torch.no_grad():
            pred = model(batch[0])

        loss = criterion(pred, batch[1].to(device))

        loss_accum += loss.detach().cpu().item()

        y_true.append(batch[1].cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim = 0)
    y_pred = torch.cat(y_pred, dim = 0)

    input_dict = {"y_true": y_true, "y_pred": y_pred}

    return evaluator.eval(input_dict), loss_accum / (step + 1)

def main():
    parser = argparse.ArgumentParser(description='Framing Detection in Climate Change')
    parser.add_argument('--random_seed', type=int, default=1042,
                        help='Random Seed for the program')

    parser.add_argument('--device', type=str, default="cuda:0", 
                        help='Selecting running device (default:cuda:0)')

    parser.add_argument('--lr', type=float, default=2e-6,
                        help='learning rate (default: 2e-6)')

    parser.add_argument('--lm', type=str, default="bert", 
                        help='pre-trained language model')

    parser.add_argument('--model', type=str, default="FD_BASE", 
                        help='model structure to use')

    parser.add_argument('--dataset', type=str, default="./dataset_original/", 
                        help='entire dataset file path')

    parser.add_argument('--specified_label', type=str, default='None', 
                        help='label for model BERT4SIN')
 
    parser.add_argument('--fine_tuning', action='store_true', 
                        help='fine tune the weights of bert')

    parser.add_argument('--dataset_balancing', action='store_true', 
                        help='Balance the label distribution in the dataset')

    parser.add_argument('--max_len', type=int, default=256,
                        help='max length the input can take (default: 256)')

    parser.add_argument('--fold', type=int, default=1,
                        help='We do 5-fold validation, select fold number here (default: 1)')

    parser.add_argument('--ckp_path', type=str, default = '', 
                        help='further pretrained model path')

    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size for training (default: 16)')

    parser.add_argument('--epochs', type=int, default=20,
                        help='number of training epochs (default: 20)')

    parser.add_argument('--log_dir', type=str, default="./log/bert/", 
                        help='tensorboard log directory')

    parser.add_argument('--checkpoint_dir', type=str, default = './ckp/', 
                        help='directory to save checkpoint')

    args = parser.parse_args()

    print(args)

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    random.seed(args.random_seed)


    if args.model == 'FD_SIN':
        train_set = ClimateDataset('train', args.dataset, args.fold).get_single_label_dataset(args.specified_label)
        if args.dataset_balancing:
            train_set.balancing_dataset()
        valid_set = ClimateDataset('dev', args.dataset, args.fold).get_single_label_dataset(args.specified_label)
        test_set  = ClimateDataset('test', args.dataset, args.fold).get_single_label_dataset(args.specified_label)
    elif args.model == 'FD_BASE':
        train_set = ClimateDataset('train', args.dataset, args.fold)
        valid_set = ClimateDataset('dev', args.dataset, args.fold)
        test_set  = ClimateDataset('test', args.dataset, args.fold)

    if args.model == 'FD_BASE':
        evaluator = Evaluator()
    elif args.model == 'FD_SIN':
        evaluator = Evaluator(classifier='multiple')

    if args.lm == 'bert':
        tk = BertTokenizer.from_pretrained("bert-base-uncased")
    elif args.lm == 'longformer':
        tk = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")

    if args.model == 'FD_BASE':
        collator = Collator(tk, args.max_len, label=True, embeded=True)
    elif args.model == 'FD_SIN':
        collator = Collator(tk, args.max_len, label=True, embeded=False)

    train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size, collate_fn=collator)
    valid_loader = DataLoader(dataset=valid_set, batch_size=args.batch_size, collate_fn=collator)
    test_loader  = DataLoader(dataset=test_set,  batch_size=args.batch_size, collate_fn=collator)
        
    if args.model == 'FD_BASE':
        model = FD_BASE(args.lm, args.max_len, args.fine_tuning, args.ckp_path).to(args.device)
    elif args.model == 'FD_SIN':
        model = FD_SIN(args.lm, args.fine_tuning, args.ckp_path).to(args.device)

    total = sum([param.nelement() for param in model.parameters()])
    print('parameters:', total)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    if args.model == 'FD_BASE':
        criterion = nn.MSELoss()
    if args.model == 'FD_SIN':
        criterion = nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    best_valid_metric = {'P':-1, 'R':-1, 'F1':-1}
    best_ckp = None
    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        if args.model == 'FD_BASE':
            train_metric = train_0(model, train_loader, optimizer, criterion, args.device)
        elif args.model == 'FD_SIN':
            train_metric = train_1(model, train_loader, optimizer, criterion, args.device)

        print('Evaluating...')

        valid_metric, valid_loss = eval(model, valid_loader, evaluator, criterion, args.device)

        print({'Train Loss': train_metric, 'Valid Loss': valid_loss, 'Validation Metric': valid_metric})

        if valid_metric['F1'] > best_valid_metric['F1']:
            best_valid_metric = valid_metric
            if args.checkpoint_dir != '':
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_val_metric': best_valid_metric}
                best_ckp = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
                torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'checkpoint.pt'))

        scheduler.step()
            
        print(f'Best validation metric so far: {best_valid_metric}, Latest Lr: {scheduler.get_last_lr()[0]}')

    # reload and test
    model.load_state_dict(torch.load(best_ckp)['model_state_dict'])
    print('Testing...')
    name = args.dataset.strip('.').strip('/') + '_' + args.model + '_' + args.specified_label + '_' + str(args.fold)
    if args.model == 'FD_BASE':
        test_evaluator = Evaluator(name=name, dir=args.log_dir, detail=True, record_id=True, tmode='test')
    elif args.model == 'FD_SIN':
        test_evaluator = Evaluator(name=name, dir=args.log_dir, classifier='multiple', record_id=True, tmode='test')
    test_metric, test_loss = eval(model, test_loader, test_evaluator, criterion, args.device)
    print(f'Test metric: {test_metric}; Test loss: {test_loss}')


if __name__ == "__main__":
    
    main()