#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import hashlib
import os

import cv2
import glob
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a dataset for InstructPix2Pix style training."
    )
    parser.add_argument(
        "--model_id", type=str, default="sayakpaul/whitebox-cartoonizer"
    )
    parser.add_argument("--dataset_id", type=str, default="imagenette")
    parser.add_argument("--max_num_samples", type=int, default=5000)
    parser.add_argument("--data_root", type=str, default="./datasets", help="datasets_dir")
    args = parser.parse_args()
    return args


# def load_dataset(dataset_id: str, max_num_samples: int) -> tf.data.Dataset:
#     dataset = tfds.load(dataset_id, split="train")
#     dataset = dataset.shuffle(max_num_samples if max_num_samples is not None else 128)
#     if max_num_samples is not None:
#         print(f"Dataset will be restricted to {max_num_samples} samples.")
#         dataset = dataset.take(max_num_samples)
#     return dataset


def main(args):
    print("Loading initial dataset")
    color_image_paths = glob.glob("C:/Users/30661/PycharmProjects/DeepLearning282/pictures/*.jpg")

    for color_image_path in color_image_paths:
        color_image = cv2.imread(color_image_path)
        gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        original_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

        hash_image = hashlib.sha1(original_image.tobytes()).hexdigest()
        sample_dir = os.path.join(args.data_root, hash_image)
        os.makedirs(sample_dir, exist_ok=True)

        cv2.imwrite(os.path.join(sample_dir, "original_image.jpg"), original_image)
        cv2.imwrite(os.path.join(sample_dir, "colorized_image.jpg"), color_image)

    print(f"Total generated image-pairs: {len(os.listdir(args.data_root))}.")


if __name__ == "__main__":
    args = parse_args()
    main(args)
