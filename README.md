<br><br>

# Dance Dance Generation
### [[Paper]](https://arxiv.org/abs/1904.00129) 
Pytorch implementation of Dance Dance Generation: Motion Transfer for Internet Videos.

## Acknowledgments
This code is heavily built on the top of [Pix2pixHD](https://github.com/NVIDIA/pix2pixHD).
We thank the authors for their great work as well as releasing the code.


## Getting Started
### Prerequisites
- Python 2.x / Python 3.x
- Pytorch 0.4 or above
- Nvidia GPU (12G or above memory recommended)
- According CUDA + CuDNN

### Training
The two-stage motion transfer model has been trained under two resolutions:
256x256 (low res) and 512x512 (high res). Please note that the number of 
filters of the first conv layer (--ngf) is half size of the original 
pix2pixHD work for the memory limitation. Feel free to increase the number to 
further improve the generation quality if GPU memory allowed 
(haven't been fully tested).

Train low res model:
```bash
python train.py --name personal_transfer_256p --resize_or_crop none --dataroot PATH_TO_TRAIN_DATA --label_nc 75 --comb_label_nc 48 --no_instance --ngf 32 --batchSize 4 --lambda_sp 10.0
```

Train high res model:
```bash
python train.py --name personal_transfer_512p --resize_or_crop none --dataroot PATH_TO_TRAIN_DATA --label_nc 75 --comb_label_nc 48 --no_instance --netG local --ngf 16 --num_D 3 --niter 50 --niter_decay 50 --niter_fix_global 10 --load_pretrain PATH_TO_LOW_RES_MODEL --batchSize 1 --lambda_sp 10.0
```
For multi-GPU settings, simply make (for instance using 4 GPUS)
```bash
--batchSize 4 --gpu_ids 0,1,2,3
```

### Testing
Test low res model:
```bash
python test.py --name personal_transfer_256p --resize_or_crop none --dataroot PATH_TO_TEST_DATA --label_nc 75 --comb_label_nc 48 --no_instance --ngf 32
```

Test high res model:
```bash
python test.py --name personal_transfer_512p --resize_or_crop none --dataroot PATH_TO_TEST_DATA --label_nc 75 --comb_label_nc 48 --no_instance  --netG local --ngf 16
```

## Dataset
#### Coming soon ... 
Feel free to test the models using your own data or videos from YouTube.


## TODO
- Code for preprocessing YouTube videos and generating train/test data.

