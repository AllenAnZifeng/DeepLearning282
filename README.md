# Enhancing Visual Realism: Fine-Tuning InstructPix2Pix for Advanced Image Colorization (CS 282A/182)
![image](https://github.com/AllenAnZifeng/DeepLearning282/blob/main/stages.png)

## Training data

https://huggingface.co/datasets/annyorange/colorized_people-dataset

## Quickstart
The code need a gpu device to run.

Requires Huggingface and wandb accounts.

Requires torch version 2.1.0+cu118 or higher and xformers to run.

Note: Installing xformers with 'pip install xformers' may replace the previously installed torch version with the CPU version. Please ensure that the torch version remains compatible with CUDA (cu118) by reinstalling the appropriate torch version if needed.

### WINDOWS
login into huggingface
```
> huggingface-cli login    
```
Confirm that you successfully logged in to huggingface
```
> huggingface-cli whoami 
```
In finetune_instruct_pix2pix.py 

```
parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
```
replace the `default` from `None` to your huggingface access token.

run the program
```
> ./run_finetune.bat   
```
in run_finetune.bat change "OUTPUT_DIR" into your path.

### LINUX
login into huggingface
```
huggingface-cli login    
```
Confirm that you successfully logged in to huggingface
```
huggingface-cli whoami 
```

In finetune_instruct_pix2pix.py 

```
parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub.",
    )
```
replace the `default` from `None` to your huggingface access token.

run the program
```
./run_finetune.sh
```
in run_finetune.sh change "OUTPUT_DIR" into your path.

## Building your own datasets or finetuning work

Make sure you have already logged in huggingface before you start running generate_dataset.py and export_to_hub.py.

By running generate_dataset.py and export_to_hub.py can help you generate your own dataset.

## testing the baseline(instructpix2pix)

```
> python ori_instructpix2pix.py
```
## loss

<p float="left">
  <img src="https://github.com/AllenAnZifeng/DeepLearning282/blob/main/W%26B%20Chart%202023_11_29%2023_04_53.png" width="400" />
  <img src="https://github.com/AllenAnZifeng/DeepLearning282/blob/main/W%26B%20Chart%202023_11_29%2023_05_08.png" width="400" />
</p>
