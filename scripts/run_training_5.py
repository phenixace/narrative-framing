import os

LABELS = ['ar', 'hi', 'co', 'mo', 'ec']

for i in range(1,6):
    for label in LABELS:
        os.system("python ../training_base.py --lm longformer --model FD_SIN --dataset ./dataset_original/ --specified_label {} --fine_tuning --dataset_balancing --max_len 256 --fold {} --log_dir ./log/longformer_o_b/ > ./log/longformer_o_b/{}_{}.txt".format(label,i,label,i))
