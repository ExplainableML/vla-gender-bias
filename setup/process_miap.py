import os
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MIAP dataset')
    parser.add_argument("--path", type=str, required=True, help="Path to MIAP dataset")
    args = parser.parse_args()

    miap_path = args.path

    # Load annotations
    test_annotations = pd.read_csv(os.path.join(miap_path, "open_images_extended_miap_boxes_test.csv"))
    val_annotations = pd.read_csv(os.path.join(miap_path + "open_images_extended_miap_boxes_val.csv"))

    # Add split to annotations
    test_annotations["split"] = "test"
    val_annotations["split"] = "validation"

    # Concatenate annotations
    annotations = pd.concat([test_annotations, val_annotations])

    # Select relevant columns
    relevant_columns = ["ImageID", "GenderPresentation", "AgePresentation", "XMin", "XMax", "YMin", "YMax", "split"]
    annotations = annotations[relevant_columns]

    # Drop images without gender annotation
    annotations = annotations[annotations["GenderPresentation"] != "Unknown"]
    annotations = annotations[annotations["GenderPresentation"].notna()]

    # Rename columns
    annotations = annotations.rename(columns={"ImageID": "file", "GenderPresentation": "gender", "AgePresentation": "age"})

    # Rename age categories
    annotations["age"] = annotations["age"].apply(lambda x: x.lower())
    # Replace unknown age with nan
    annotations["age"] = annotations["age"].apply(lambda x: x if x != "unknown" else None)
    # Rename gender categories
    annotations["gender"] = annotations["gender"].map({"Predominantly Feminine": "female", "Predominantly Masculine": "male"})

    # Reset index
    annotations = annotations.reset_index(drop=True)

    # Filter non-existing imagesfilenames
    missing_or_faulty_image_indices = []
    image_sizes = []  # Additionally collect image sizes
    for index, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Checking for missing images"):
        file_name = row["file"]
        split = row["split"]
        file_path = os.path.join(miap_path, split, file_name + ".jpg")
        # Remove missing images
        if not os.path.exists(file_path):
            missing_or_faulty_image_indices.append(index)
            continue
        # Remove images that cannot be opened by PIL
        try:
            image = Image.open(file_path)
        except:
            missing_or_faulty_image_indices.append(index)
            continue
    
        x_min, x_max, y_min, y_max = row[["XMin", "XMax", "YMin", "YMax"]]
        # Remove images where we cannot extract the bounding box (likely truncated image files)
        try:
            image = image.crop((x_min * image.width, y_min * image.height, x_max * image.width, y_max * image.height))
        except:
            missing_or_faulty_image_indices.append(index)
            continue
        width, height = image.width, image.height
        image_sizes.append(
            {
                "file": file_name,
                "split": split,
                "width": width,
                "height": height
            }
        )

    print(f"Removing {len(missing_or_faulty_image_indices)} missing images")
    annotations = annotations.drop(missing_or_faulty_image_indices, axis=0)

    # Map filenames to image sizes
    image2width = {entry["file"]: entry["width"] for entry in image_sizes}
    image2height = {entry["file"]: entry["height"] for entry in image_sizes}

    # Add width and height to annotations
    annotations["width"] = annotations["file"].map(image2width)
    annotations["height"] = annotations["file"].map(image2height)

    # Map filenames to image sizes
    image2width = {entry["file"]: entry["width"] for entry in image_sizes}
    image2height = {entry["file"]: entry["height"] for entry in image_sizes}

    # Add width and height to annotations
    annotations["width"] = annotations["file"].map(image2width)
    annotations["height"] = annotations["file"].map(image2height)

    # Reset index
    annotations = annotations.reset_index(drop=True)

    # Group files by gender
    grouped_entries = defaultdict(list)
    for row_index, row in annotations.iterrows():
        grouped_entries[row["gender"]].append(row_index)

    # Get median image size
    median_width = annotations["width"].median()
    median_height = annotations["height"].median()

    # Order by crop size in descending order
    def image_size(index: int) -> float:
        filename = annotations.loc[index, "file"]
        return abs(image2width[filename] * image2height[filename])

    grouped_entries = {group: list(sorted(indices, key=image_size, reverse=True)) for group, indices in grouped_entries.items()}

    # Interleave groups
    interleaved_image_indices = []
    all_groups = list(sorted(grouped_entries.keys()))
    while any(grouped_entries.values()):
        for group in all_groups:
            if grouped_entries[group]:
                interleaved_image_indices.append(grouped_entries[group].pop(0))

    # Reorder annotations
    reordered_annotations = annotations.loc[interleaved_image_indices]
    reordered_annotations = reordered_annotations.reset_index(drop=True)

    # Save images
    target_path = os.path.join(miap_path, "images")
    os.makedirs(target_path, exist_ok=True)

    image_names = []
    for row_index, row in tqdm(reordered_annotations.iterrows(), total=len(reordered_annotations), desc="Saving MIAP images"):
        # Load image
        file_name = row["file"]
        split = row["split"]
        source_path = os.path.join(miap_path, split, file_name + ".jpg")
        image = Image.open(source_path).convert("RGB")
    
        # Extract bounding box
        x_min, x_max, y_min, y_max = row[["XMin", "XMax", "YMin", "YMax"]]
        image = image.crop((x_min * image.width, y_min * image.height, x_max * image.width, y_max * image.height))
    
        # Save image
        image_name = f"miap-{row['gender']}-none-{row['age']}-{row_index}.png"
        image_names.append(image_name)
        image.save(os.path.join(target_path, image_name), format="PNG")

    # Update paths in annotations
    reordered_annotations["name"] = image_names

    # Remove crop coordinates from annotations
    reordered_annotations = reordered_annotations.drop(["XMin", "XMax", "YMin", "YMax", "width", "height", "file"], axis=1)

    # Save annotations
    reordered_annotations.to_csv(os.path.join(miap_path, "annotations.csv"), index=False)
