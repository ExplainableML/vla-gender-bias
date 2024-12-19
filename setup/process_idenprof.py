import os
import shutil
import argparse
import pandas as pd

from tqdm import tqdm

if __name__ == '__main__':
    # Parse path to IdenProf dataset
    parser = argparse.ArgumentParser("Process IdenProf dataset")
    parser.add_argument("--idenprof-path", type=str, required=True, help="Path to the IdenProf dataset")
    parser.add_argument("--output-path", type=str, required=True, help="Path to the output directory")
    args = parser.parse_args()

    # Get all directories in the dataset
    dataset_path = args.idenprof_path
    directories = os.listdir(dataset_path)
    directories_full_path = [os.path.join(dataset_path, d) for d in directories]
    directories_full_path = [d for d in directories_full_path if os.path.isdir(d)]

    # Make sure there are only 2 directories, namely 'train' and 'test'
    assert len(directories) == 2, "There should be exactly 2 directories in the dataset"
    assert 'train' in directories, "There should be a 'train' directory in the dataset"
    assert 'test' in directories, "There should be a 'test' directory in the dataset"

    # Merge splits and copy images
    output_path = args.output_path
    assert os.path.exists(output_path), "Output path does not exist"
    images_path = os.path.join(output_path, 'images')
    os.makedirs(images_path, exist_ok=True)

    annotations = []
    progress_bar = tqdm(desc="Processing IdenProf", total=11000)
    for split in ['train', 'test']:
        split_path = os.path.join(dataset_path, split)
        for occupation in os.listdir(split_path):
            occupation_path = os.path.join(split_path, occupation)
            if not os.path.isdir(occupation_path):
                continue

            for image in os.listdir(occupation_path):
                if not image.endswith('.jpg'):
                    continue
                
                image_id = image.split('-')[1].split(".")[0]
                target_name = f"{split}_{occupation}_{image_id}.jpg"
                target_path = os.path.join(images_path, target_name)
                shutil.copy(os.path.join(occupation_path, image), target_path)
                annotations.append(
                    {
                        "name": target_name,
                        "occupation": occupation,
                        "split": split
                    }
                )
                progress_bar.update(1)
        
    progress_bar.close()

    # Save annotations
    annotations = pd.DataFrame(annotations)
    annotations_path = os.path.join(output_path, 'annotations.csv')
    annotations.to_csv(annotations_path, index=False)

    # Remove original dataset
    shutil.rmtree(dataset_path)
