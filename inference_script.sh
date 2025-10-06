#!/bin/bash
python inference.py --task_id 0 --model_dir ./model_custom/wifi/6_27_test\
 --cond_dir ./dataset/wifi/raw --out_dir ./dataset/wifi/output/6_27_test\
  --data_key cfr --cond_key None --if_log True