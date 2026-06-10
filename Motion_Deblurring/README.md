### Training
##### GOPRO

 1. Download the training and testing dataset
 2. Generate image patches from full-resolution training images of GOPRO dataset, run
```python generate_patches_gopro.py```
 1. To train StarIR on the GOPRO dataset in two stages, run
```
cd StarIR
CONFIG=Desnowing/Options/GOPRO.yml python train_StarIR.py 
 ```
 after finishing, then run (do not forget to set the pretrain_network_g in GOPRO-S2.yml)
 ```
cd StarIR
CONFIG=Desnowing/Options/GOPRO-S2.yml python train_StarIR.py 
 ```

##### RealBlur

1. Download the training and testing dataset

2. To train StarIR on RealBlur-J, run
```
cd StarIR
CONFIG=Desnowing/Options/RealBlur-J.yml python train_StarIR.py 
 ```
on RealBlur-R, run
```
cd StarIR
CONFIG=Desnowing/Options/RealBlur-R.yml python train_StarIR.py 
 ```

### Evaluation
##### GOPRO/HIDE

1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```
python evaluation_from_data.py --data GOPRO
```
```
python evaluation_from_data.py --data HIDE
``` 
to generate resulting images
3. Use the MATLAB (evaluate_gopro_hide.m) to obtain PSNR/SSIM scores
##### RealBlur-J/RealBlur-R

1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```
python evaluation_from_data.py --data RealBlur-J
```
```
python evaluation_from_data.py --data RealBlur-R
``` 
3. To obtain PSNR/SSIN scores, run
```
python evaluate_realblur.py
```

