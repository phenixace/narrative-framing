import os

LABELS = ['ar', 'hi', 'co', 'mo', 'ec']


for i in range(1,6):
    for label in LABELS:
        os.system("python ../training_rbf.py --random_seed 1042 --epoch 20 --dataset ./dataset_sentence/ --specified_label {} --fine_tuning --dataset_balancing --n_passages 4 --fold {} --log_dir ./log/stb_o_b_4/ > ./log/stb_o_b_4/{}_{}.txt".format(label,i,label,i))
