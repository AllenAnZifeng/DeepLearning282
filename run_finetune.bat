set MODEL_ID=timbrooks/instruct-pix2pix
set DATASET_ID=annyorange/colorized-dataset
set OUTPUT_DIR=colorization-finetuned

accelerate launch --mixed_precision="fp16" finetune_instruct_pix2pix.py ^
  --pretrained_model_name_or_path=%MODEL_ID% ^
  --dataset_name=%DATASET_ID% ^
  --use_ema ^
  --enable_xformers_memory_efficient_attention ^
  --resolution=256 --random_flip ^
  --train_batch_size=2 --gradient_accumulation_steps=4 --gradient_checkpointing ^
  --max_train_steps=1000 ^
  --checkpointing_steps=10 --checkpoints_total_limit=50 ^
  --learning_rate=5e-05 --lr_warmup_steps=0 ^
  --mixed_precision=fp16 ^
  --val_image_url="https://raw.githubusercontent.com/AllenAnZifeng/DeepLearning282/main/how-ot-make-picture-black-and-white-after.jpg" ^
  --validation_prompt="Layer colors onto the image" ^
  --seed=42 ^
  --output_dir=%OUTPUT_DIR% ^
  --report_to=wandb ^
  --push_to_hub