### Training 
 1. Download the training and testing dataset
 2. To train StarIR, run
```
cd StarIR
CONFIG=Low_Light_Deblurring/Options/LOL-Blur.yml python train_StarIR.py
```

### Evaluation
1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```python evaluation_from_data.py``` to generate resulting images
3. Run ```python calculate_psnr_ssim.py ``` 
to obtain the numerical results
