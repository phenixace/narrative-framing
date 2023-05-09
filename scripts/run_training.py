import os

LABELS = ['ar', 'hi', 'co', 'mo', 'ec']

for i in range(1,6):
    for label in LABELS:
        os.system("python ../training_base.py --model FD_SIN --dataset ./dataset_processed_raw/ --specified_label {} --fine_tuning --dataset_balancing --max_len 256 --fold {} --log_dir ./log/bert_o_b/ > ./log/bert_o_b/{}_{}.txt".format(label,i,label,i))
