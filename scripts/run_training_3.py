import os

LABELS = ['ar', 'hi', 'co', 'mo', 'ec']

for i in range(1,6):
    for label in LABELS:
        os.system("python ../training_rbf.py --random_seed 1042 --epoch 20 --dataset ./dataset_sentence/ --specified_label {} --fine_tuning --dataset_balancing --n_passages 5 --fold {} --log_dir ./log/stb_o_b_5/ > ./log/stb_o_b_5/{}_{}.txt".format(label,i,label,i))

