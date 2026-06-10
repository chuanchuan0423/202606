### Training
 1. Download the training and testing dataset
 2. To train StarIR on the AGAN dataset, run
```
cd StarIR
CONFIG=Deraining/Options/AGAN.yml python train_StarIR.py
 ```

on the SPAD dataset, run

```
cd StarIR
CONFIG=Deraining/Options/SPAD.yml python train_StarIR.py
 ```

on the DID dataset, run 
```
cd StarIR
CONFIG=Deraining/Options/DID.yml python train_StarIR.py
 ```



on the Rain13k dataset, run

```
cd StarIR
CONFIG=Deraining/Options/Rain13k.yml python train_StarIR.py
 ```

### Evaluation
**For SPAD/AGAN**

1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```
python evaluation_from_data.py --data AGAN
python evaluation_from_data.py --data SPAD
``` 
to generate resulting images

(evaluation is performed on the test_a subset of AGAN)

3. Run 
```
python calculate_psnr_ssim.py --data AGAN --test_y_channel
python calculate_psnr_ssim.py --data SPAD --test_y_channel
``` 
to obtain the numerical results

**For Rain13k**
1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```
python evaluation_rain13k.py
``` 
3. Obtain the PSNR/SSIM scores using the MATLAB file