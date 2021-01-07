An implementation of cycle-gan that trains on [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Using wgan with gradient penalty to stabilise training and avoid model collapse.

## Run code
install requirements:
```shell script
git clone https://github.com/MorvanZhou/celebA-cyclegan
cd celebA-cyclegan
pip3 install -r requirements.txt
```

Download CelebA img_align_celeba.zip (~1.4GB) from [https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing),
and list_attr_celeba.txt (25MB) from [https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing](https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing).

Parse data
```shell script
python dataset.py --data_dir ~/data/celebA_img_align/
```

Training
```shell script
python train.py --data_dir ~/data/celebA_img_align/ --soft_gpu -b 32 --epoch 51 --cycle_lambda 10 --gp_lambda 10 -lr 0.0005 -b1 0.01 -b2 0.99
```

Test
```shell script
python restore.py --model_path visual\2020-12-17_16-06-29\model\cp-0020-00002000.ckpt -t f2m --image_path demo/female.jpg
```

![image](/demo/ep049t1000.png)
