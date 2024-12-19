import os
import json
import shutil
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from collections import Counter
from collections import defaultdict


if __name__ == '__main__':
    # Parse path
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    # Load annotations
    fairface_path = args.path
    train_annotations = pd.read_csv(os.path.join(fairface_path, "fairface_label_train.csv"))
    val_annotations = pd.read_csv(os.path.join(fairface_path, "fairface_label_val.csv"))
    annotations = pd.concat([train_annotations, val_annotations])

    # Extract relevant columns
    annotations = annotations[["file", "gender", "race", "age"]]

    # Rename race to ethnicity
    annotations = annotations.rename(columns={"race": "ethnicity"})

    # Normalize attributes
    annotations["gender"] = annotations["gender"].apply(lambda x: x.lower())
    annotations["ethnicity"] = annotations["ethnicity"].apply(lambda x: x.lower().replace(" ", "_"))

    # Replace latino_hispanic with latino
    annotations["ethnicity"] = annotations["ethnicity"].apply(lambda x: "latino" if x == "latino_hispanic" else x)

    # Remove images where age is < 20 or > 70
    annotations = annotations[~annotations["age"].isin(["0-2", "3-9", "10-19", 'more than 70'])]

    # Check for missing or faulty data
    faulty_image_indices = []
    for i, row in tqdm(annotations.iterrows(), total=len(annotations)):
        for margin in ["margin025", "margin125"]:
            # Make source path
            source_path = os.path.join(fairface_path, margin)
            source_file = os.path.join(source_path, row["file"])
            try:
                img = Image.open(source_file).convert("RGB")
            except:
                faulty_image_indices.append(i)
    
    # Remove faulty images
    num_images_before_removal = len(annotations)
    annotations = annotations.drop(faulty_image_indices)
    num_images_after_removal = len(annotations)
    print(f"Removed {num_images_before_removal - num_images_after_removal} faulty images")

    # Convert annotations to records
    annotations = annotations.to_dict("records")

    # Group files by combination of gender and ethnicity
    grouped_files = defaultdict(list)
    for record in annotations:
        grouped_files[(record["gender"], record["ethnicity"])].append(record)

    all_groups = list(sorted(grouped_files.keys()))
    reordered_annotations = []

    while any(grouped_files.values()):
        for group in all_groups:
            if grouped_files[group]:
                reordered_annotations.append(grouped_files[group].pop(0))

    reordered_annotations = pd.DataFrame(reordered_annotations)

    # Save reordered annotations
    for margin in ["margin025", "margin125"]:
        source_path = os.path.join(fairface_path, margin)
        target_path = os.path.join(fairface_path, "..", f"fairface_{margin}", "images")
        os.makedirs(target_path, exist_ok=True)
        image_names = []

        for row_index, row in tqdm(reordered_annotations.iterrows(), total=len(annotations), desc=f"Copying {margin} images"):
            file_suffix = row["file"]
            # Split off file name
            image_name = f"fairface_{margin}-{row['gender']}-{row['ethnicity']}-{row['age']}-{row_index}.jpg"
            source_file = os.path.join(source_path, row["file"])
            target_file = os.path.join(target_path, image_name)
            shutil.copyfile(source_file, target_file)
            image_names.append(image_name)
            
            assert os.path.exists(os.path.join(target_path, image_name)), f"File {target_file} does not exist"

    # Update paths in annotations
    reordered_annotations["name"] = image_names

    # Save annotations
    reordered_annotations_margin025 = reordered_annotations.copy()
    reordered_annotations_margin025["name"] = reordered_annotations_margin025["name"].apply(lambda x: x.replace("125", "025"))

    reordered_annotations_margin125 = reordered_annotations.copy()
    reordered_annotations_margin025.to_csv(
        os.path.join(fairface_path, "..", "fairface_margin025", "annotations.csv"),
        index=False
    )
    reordered_annotations_margin125.to_csv(
        os.path.join(fairface_path, "..", "fairface_margin125", "annotations.csv"),
        index=False
    )
