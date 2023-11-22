from PIL import Image
import glob

# Convert image to black and white, still use RGB channels
def convert_to_bw(image: Image.Image):
    return image.convert("L").convert("RGB")


if __name__ == "__main__":
    output_folder = "processed_dataset"
    
    for f in glob.glob("dataset/*.png"):
        image = Image.open(f)
        image = convert_to_bw(image)
        image.save(f.replace("dataset", output_folder))
