#!/bin/bash
# nohup python train.py --task_id 0 --model_dir ./model_custom/wifi/6_27_test2\
#  --log_dir ./log/wifi/6_27_test2 --data_key cfr --cond_key None > 6_27_test2.out 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --task_id 12 > RunInfo/exp_vivo_10_5_FT20.out 2>&1 & 