from PIL import Image
import glob
from datasets import load_dataset
import os

# Convert image to black and white, still use RGB channels
def convert_to_bw(image: Image.Image):
    return image.convert("L").convert("RGB")


def test():
    data_files = {}
    data_files["train"] = os.path.join("dataset/train", "**")
    
    dataset = load_dataset(
        "dataset",
        data_files=data_files
    )
    print(dataset["train"].column_names)
    print(dataset['train'][1])



def main():
    output_folder = "processed_dataset"
    
    for f in glob.glob("dataset/*.png"):
        image = Image.open(f)
        image = convert_to_bw(image)
        image.save(f.replace("dataset", output_folder))


if __name__ == "__main__":
    test()
