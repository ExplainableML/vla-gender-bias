import os
import json
import argparse
import pandas as pd

from tqdm import tqdm
from PIL import Image
from collections import defaultdict, Counter


def extract_annotations_for_person(person_annotations: list[dict]) -> dict:
    annotations = defaultdict(list)
    for annotation in person_annotations:
        for entry_key, entry_value in annotation.items():
            if entry_key == "worker":
                continue
            annotations[entry_key].append(entry_value)
    
    # Extract most common value for each attribute
    # If there is no unique most common value, return None
    voted_annotations = {}
    for attribute, values in annotations.items():
        attribute_value_counter = Counter(values)
        counts = list(attribute_value_counter.values())
        maximum_count = max(counts)
        if counts.count(maximum_count) > 1:
            voted_annotations[attribute] = None
        else:
            voted_annotations[attribute] = max(attribute_value_counter, key=attribute_value_counter.get)
    
    return voted_annotations


def extract_annotations_for_image(image_data: dict, image_name: str, split: str) -> tuple[list[dict], dict[str, int]]:
    error_counts = defaultdict(int)
    # First, load the image
    try:
        image_path = os.path.join(phase_path, split, image_name)
        image = Image.open(image_path).convert("RGB")
    except:
        error_counts["faulty_image"] += 1
        return []
    
    image_annotations = []
    for person_key, person_annotations in image_data.items():
        # Ignore caption
        if person_key == "caption":
            error_counts["is_caption"] += 1
            continue
        # Skip images without bounding box annotations
        if "region" not in person_annotations:
            error_counts["no_region"] += 1
            continue
        # Skip images without person annotations
        if "annotations" not in person_annotations:
            error_counts["no_annotations"] += 1
            continue
        
        bounding_box = person_annotations["region"]
        attribute_annotations = person_annotations["annotations"]
        person_attribute_annotations = extract_annotations_for_person(attribute_annotations)
        person_attribute_annotations["bounding_box"] = bounding_box
        
        # Crop the bounding box
        try:
            person_crop = image.crop(bounding_box)
            person_attribute_annotations["image"] = person_crop
        except:
            error_counts["faulty_crop"] += 1
            continue
        
        # Add image name to annotations
        person_attribute_annotations["image_name"] = image_name
        # Add split to annotations
        person_attribute_annotations["split"] = split
        
        # Save the annotations
        image_annotations.append(person_attribute_annotations)
    
    return image_annotations, error_counts


def image_size(image: Image.Image) -> float:
    width = image.width
    height = image.height
    return abs(width * height)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    phase_path = args.path

    # Load annotations
    train_annotations_path = os.path.join(phase_path, "phase_gcc_train_all_20221101.json")
    validation_annotations_path = os.path.join(phase_path, "phase_gcc_val_all_20221101.json")

    with open(train_annotations_path, "r") as file:
        train_annotations = json.load(file)
    
    with open(validation_annotations_path, "r") as file:
        validation_annotations = json.load(file)

    annotations_with_split = [(train_annotations, "train"), (validation_annotations, "val")]
    all_person_annotations = []
    all_error_counts = defaultdict(int)
    for split_annotations, split in annotations_with_split:
        # Extract annotations
        progress_bar = tqdm(split_annotations.items(), desc=f"Extracting {split} annotations")
        for image_name, image_data in progress_bar:
            image_annotations, error_counts = extract_annotations_for_image(image_data, image_name, split)
            all_person_annotations.extend(image_annotations)
            for error_key, error_count in error_counts.items():
                all_error_counts[error_key] += error_count
        
            progress_bar.set_postfix({"Number of annotations": len(all_person_annotations)})
    
    for error_key, error_count in all_error_counts.items():
        print(f"Removed {error_count} annotations due to {error_key}")
    
    ages = list(set([person["age"] for person in all_person_annotations]))
    genders = list(set([person["gender"] for person in all_person_annotations]))
    ethnicities = list(set([person["ethnicity"] for person in all_person_annotations]))

    # Show num. images before filtering
    print(f"Number of images before filtering: {len(all_person_annotations)}")

    # Remove images with unknown gender
    all_person_annotations = [person for person in all_person_annotations if person["gender"] in ["male", "female"]]

    # Remove images with unknown or non-standard ethnicity
    all_person_annotations = [person for person in all_person_annotations if person["ethnicity"] not in [None, "unsure", "other"]]

    # Remove images of non-adults
    all_person_annotations = [person for person in all_person_annotations if person["age"] in ["adult", "senior", "young", "unsure"]]

    # Remove images with activity in {music, sports}
    all_person_annotations = [person for person in all_person_annotations if person.get("activity", None) not in ["music", "sports"]]

    # Show num. images after filtering
    print(f"Number of images after filtering: {len(all_person_annotations)}")

    # Group images by combinations of gender and ethnicity
    grouped_images = defaultdict(list)
    for person in all_person_annotations:
        gender = person["gender"]
        ethnicity = person["ethnicity"]
        grouped_images[(gender, ethnicity)].append(person)
    
    # Within each group, sort images by crop size in descending order
    grouped_images_sorted = {group: list(sorted(images, key=lambda x: image_size(x["image"]), reverse=True)) for group, images in grouped_images.items()}


    # Save images
    annotations = []

    all_groups = list(sorted(grouped_images_sorted.keys()))
    pbar = tqdm(total=len(all_person_annotations), desc="Saving phase images")
    while any(grouped_images_sorted.values()):
        for group in all_groups:
            if grouped_images_sorted[group]:
                person = grouped_images_sorted[group].pop(0)
                image = person["image"]
                image_name = f"phase-{person['gender']}-{person['ethnicity']}-{person.get('age', 'none')}-{len(annotations)}.png"
                person["name"] = image_name
            
                save_path = os.path.join(phase_path, "images")
                os.makedirs(save_path, exist_ok=True)
                image.save(os.path.join(save_path, image_name), format="PNG")
                annotations.append({key: value for key, value in person.items() if key != "image"})
                pbar.update(1)
                pbar.refresh()

    pbar.close()

    annotations = pd.DataFrame(annotations)

    # Remove bounding box from annotations
    annotations = annotations.drop("bounding_box", axis=1)

    # Remove split from annotations
    annotations = annotations.drop("split", axis=1)

    # Reorder columns
    annotations = annotations[["name", "gender", "ethnicity", "age", "activity", "emotion"]]

    # Save annotations
    annotations.to_csv(os.path.join(phase_path, "annotations.csv"), index=False)
