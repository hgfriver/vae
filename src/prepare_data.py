import glob as glob
import os
from tqdm import tqdm
def prepare_dataset(ROOT_PATH):
    image_dirs = os.listdir(ROOT_PATH)
    image_dirs.sort()
    print(len(image_dirs))
    print(image_dirs[:5])
    all_image_paths = []
    for i in tqdm(range(len(image_dirs))):
        image_paths = glob.glob(f"{ROOT_PATH}/{image_dirs[i]}/*")
        image_paths.sort()
        for image_path in image_paths:
            all_image_paths.append(image_path)
        
    print(f"Total number of face images: {len(all_image_paths)}")
    train_data = all_image_paths[:-2000]
    valid_data = all_image_paths[-2000:]
    print(f"Total number of training image: {len(train_data)}")
    print(f"Total number of validation image: {len(valid_data)}")
    return train_data, valid_data
