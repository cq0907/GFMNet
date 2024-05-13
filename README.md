# Global-to-Local Feature Mining Network for RGB-Infrared Person ReIdentification (MMM 2024) 
Pytorch Code of our method [1] for Cross-Modality Person Re-Identification (Visible Thermal Re-ID) on RegDB dataset [2] and SYSU-MM01 dataset [3]. 

|Datasets    | Pretrained| Rank@1  | mAP |  mINP |  
| --------   | -----    | -----  |  -----  | ----- |
|#RegDB      | ImageNet | ~ 87.14% | ~ 80.65%|  ~66.83% |
|#SYSU-MM01  | ImageNet | ~ 73.04%  | ~ 69.71% | ~56.62% | 

*Both of these two datasets may have some fluctuation due to random spliting. The results might be better by finetuning the hyper-parameters. 

### 1. Prepare the datasets.

- (1) RegDB Dataset [2]: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

- (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

- A private download link can be requested via sending me an email (mangye16@gmail.com). 

- (2) SYSU-MM01 Dataset [3]: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

- run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

### 2. Training (Section 4.2).
Train a model by
```bash
python train_ext_test0.py --dataset sysu --lr 0.1 --method adp --augc 1 --rande 0.5 --alpha 1 --square 1 --gamma 1 --gpu 1
```

- `--dataset`: which dataset "sysu" or "regdb".

- `--lr`: initial learning rate.

-  `--method`: method to load loss functions.

-  `--augc`:  Channel augmentation or not.

-  `--rande`:  random erasing with probability.

- `--gpu`:  which gpu to run.

You may need mannully define the data path first.

### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset by using testing augmentation with HorizontalFlip
```bash
python testa.py --mode all --resume 'model_path' --gpu 1 --dataset sysu
```
- `--dataset`: which dataset "sysu" or "regdb".

- `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).

- `--trial`: testing trial (only for RegDB dataset).

- `--resume`: the saved model path.

- `--gpu`:  which gpu to run.

For example: python testa.py --mode all --gpu 1 --dataset sysu --resume sysu_adp_joint_co_nog_ch_nog_sq1_aug_G_erase_0.5_p4_n8_lr_0.1_seed_0_9_best.t

[comment]: <> (### 4. Visualization)

[comment]: <> (Visualization of heat map)

[comment]: <> (```bash)

[comment]: <> (python visualization2.py --dataset sysu)

[comment]: <> (```)

[comment]: <> (Visualization of search results)

[comment]: <> (```bash)

[comment]: <> (python visualization_sort.py --dataset sysu)

[comment]: <> (```)

### 4. Citation

Please kindly cite this paper in your publications if it helps your research:
```
@inproceedings{iccv21caj,
author    = {Chen, Qiang and He, Fuxiao and Xiao, Guoqiang},
title     = {Global-to-Local Feature Mining Network for RGB-Infrared Person ReIdentification},
booktitle = {International Conference on Multimedia Modeling},
year      = {2024},
pages     = {1-13}
}
```

###  5. References.
[1] Q. Chen, F. X. He, G. Q. Xiao. Global-to-Local Feature Mining Network for RGB-Infrared Person Re-Identification[C], International Conference on Multimedia Modeling. 2024: 1-13.

[2] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

[3] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.
