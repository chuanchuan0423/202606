

## Installation and Data Preparation

See [INSTALL.md](INSTALL.md) for the installation of dependencies and dataset preperation required to run this codebase.

## Training

After preparing the training data in ```data/``` directory, use 
```
cd All_in_One
python train.py
```
to start the training of the model. Use the ```de_type``` argument to choose the combination of degradation types to train on. By default it is set to all the 5 degradation tasks (denoising, deraining, dehazing, deblurring, enhancement).

Example Usage: If we only want to train on deraining and dehazing:
```
python train.py --de_type derain dehaze
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. To perform the evaluation, use
```
cd All_in_One
python test.py --mode {n} 
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehazing, 3 for deblurring, 4 for enhancement, 5 for three-degradation all-in-one setting and 6 for five-degradation all-in-one setting.

Example Usage: To test on all the degradation types at once, run:

```
python test.py --mode 6 --ckpt_name StarIR-AIO-5D.ckpt
```

## Demo


To obtain visual results from the model, ``demo.py`` can be used. After placing the pre-trained models of [three-task](https://drive.google.com/file/d/1hU0sXyhJb1MgPIlGLnQRtY-RFm6Qb641/view?usp=sharing) or [five-task](https://drive.google.com/file/d/1vG0M7_OpTi8afivxzacur4T-k5abptYa/view?usp=sharing) settings in ``ckpt`` directory, run:

```
cd All_in_One
python demo.py --ckpt_name StarIR-AIO-5D.ckpt --test_path {path_to_degraded_images} --output_path {save_images_here} 
```

Example usage to run inference on a directory of images:
```
cd All_in_One
python demo.py --ckpt_name StarIR-AIO-5D.ckpt  --test_path './demo/degraded/' --output_path './demo/restored/'
```

Example usage to run inference on an image directly:

```
cd All_in_One
python demo.py --ckpt_name StarIR-AIO-5D.ckpt  --test_path './demo/degraded/1.jpg' --output_path './demo/degraded/'
```

To use tiling option while running ``demo.py`` set ``--tile`` option to ``True``. The Tile size and Tile overlap parameters can be adjusted using ``--tile_size`` and ``--tile_overlap`` options respectively.
