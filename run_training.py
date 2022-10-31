import os

DATASET = ['./dataset_sentence/']
LABELS = ['ar', 'hi', 'mo', 'co', 'ec']



for dt in DATASET:
    for i in range(1,6):
        for label in LABELS:
            for j in range(2,6):
                os.system("python training_stb.py --fusion linear --lr 2e-6 --lm bert --dataset {} --fine_tuning --max_len 64 --fold {} --specified_label {} --n_passages {} --log_dir ./log/linear/stb-ub-{}/ > ./log/linear/stb-ub-{}/{}_{}_{}_{}.txt".format(dt, str(i), label, str(j), str(j), str(j), dt.strip('.').strip('/'), str(i), label, str(j)))

'''
for dt in DATASET:
    for i in range(1,6):
        os.system("python training_base.py --model FD_BASE --ckp_path --lr 2e-6 --lm bert --dataset {} --fine_tuning --max_len 256 --fold {} --log_dir ./log/f_bert-256-ub-s/ > ./log/f_bert-256-ub-s/{}_{}_{}.txt".format(dt, str(i), label, dt.strip('.').strip('/'), str(i), label))
'''