import os

LABELS = ['ar', 'hi', 'co', 'mo', 'ec']

for i in range(1,6):
    for label in LABELS:
        os.system("python ./experiments/training_baseline.py --model FD_SIN --dataset ./experiments/data_splits_raw/ --specified_label {} --fine_tuning --dataset_balancing --max_len 256 --fold {} --log_dir ./log/".format(label,i))
