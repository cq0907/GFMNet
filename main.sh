#!/usr/bin/env bash

## Train
#python train_afmm.py --dataset sysu --lr 0.1 --method adp --augc 1 --rande 0.5 --alpha 1 --square 1 --gamma 1 --gpu 1

# Test
python testa.py --mode all --dataset sysu --resume sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n 8_lr_0.1_seed_0_9_best.t --gpu 1