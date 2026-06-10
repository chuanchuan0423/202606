### Training
 1. Download the training and testing dataset
 2. To train StarIR on the ITS dataset, run
```
cd StarIR
CONFIG=Desnowing/Options/ITS.yml python train_StarIR.py
 ```


### Evaluation
1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```
python evaluation_from_data.py --data ITS
``` 
to generate resulting images
3. Run 
```
python calculate_psnr_ssim.py --data ITS
``` 
to obtain the numerical results

``ITS`` can be replaced by ``OTS``, ``Haze4k``, ``Dense-Haze``, ``NH-HAZE``,  
``SateHaze1k-Thin``, ``SateHaze1k-Moderate``, ``SateHaze1k-Thick``, ``RESIDE-6K`` for other datasets

