### Training 
 1. Download the training and testing dataset
 2. Generate image patches from full-size training images, run
 ```
 python generate_patches_dpdd.py
 ```
 1. To train StarIR for single-image defocus deblurring, run
```
cd StarIR
CONFIG=Defocus_Deblurring/Options/DPDD-Single.yml python train_StarIR.py
 ```
To train StarIR for dual-pixel defocus deblurring, run
```
cd StarIR
CONFIG=Defocus_Deblurring/Options/DPDD-Dual.yml python train_StarIR.py
 ```

### Evaluation
1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```python evaluate_single_dpdd.py``` 
to generate resulting images for single-image defocus deblurring
3. Run ```python evaluate_dual_dpdd.py ``` 
to obtain the numerical results for dual-pixel defocus deblurring
