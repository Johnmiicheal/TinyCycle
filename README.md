
<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# TinyCycle
An implementation of the CycleGAN image to image translation applying Monet, Vangogh, Cezanne and Ukiyo-e artistic styles to a webcam feed.

## Usage
Run using ``` python webcam.py  --name MODEL_NAME --model test --preprocess none --no_dropout ``` 

## Style Models
| Models  |
| ------------- | 
| MONET  |
| VANGOGH |
| UKIYOE  |
| CEZANNE |


## General Purpose
Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.
It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.
Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        ```python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan```
    Test a CycleGAN model (one side only):
        ```python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout```
    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.
    Test a pix2pix model:
        ```python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA```

**See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md **


**For more information on CycleGAN, and to see the wonderful work of the original creators: [Project](https://junyanz.github.io/CycleGAN/) |  [Paper](https://arxiv.org/pdf/1703.10593.pdf) |  [Torch](https://github.com/junyanz/CycleGAN) |
[Tensorflow Core Tutorial](https://www.tensorflow.org/tutorials/generative/cyclegan) | [PyTorch Colab](https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb)**
