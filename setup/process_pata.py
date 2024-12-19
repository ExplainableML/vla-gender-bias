import os
import shutil
import argparse
import requests
import pandas as pd
import urllib.request as urlopen

from tqdm import tqdm
from PIL import Image
from urllib.error import HTTPError
from collections import defaultdict
from PIL import UnidentifiedImageError


def save_image(idx, url, base_path):
    os.makedirs(base_path, exist_ok=True)
    save_path = os.path.join(base_path, f"{idx}.png")
    if os.path.exists(save_path):
        return True

    try:
        img = Image.open(requests.get(url, stream=True).raw)
    except UnidentifiedImageError:
        opener = urlopen.build_opener()
        opener.addheaders = [('User-Agent', 'Chrome')]
        urlopen.install_opener(opener)
        try:
            urlopen.urlretrieve(url, save_path)
        except HTTPError:
            return
        except UnidentifiedImageError:
            return
                    
        try:
            img = Image.open(save_path)
        except UnidentifiedImageError:
            # print(f"UnidentifiedImageError {idx}: {url}")
            os.remove(save_path)
            return
    except:
        return

    img.save(save_path, format="png")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    
    path_to_table = os.path.join(args.path, "pata_fairness.files.lst")
    table = pd.read_csv(path_to_table, sep="|", names=["label", "url"])

    progress = tqdm(enumerate(zip(table["label"].tolist(), table["url"].tolist())), total=len(table))
    errros = 0
    success = 0
    
    annotations = []

    for idx, (label, url) in progress:
        *_, ethnicity, gender, age = label.split("_")
        if ethnicity == "eastasian":
            ethnicity = "east_asian"
        
        image_name = f"{label}_{idx}.png"
        status = save_image(f"{label}_{idx}", url, os.path.join(args.path, "images_tmp"))
        if status is None:
            errros += 1
        else:
            success += 1
            annotations.append(
                {
                    "name": image_name,
                    "ethnicity": ethnicity,
                    "gender": gender,
                    "age": age,
                }
            )
    
        success_ratio = 100 * success / (errros + success)
        progress.set_postfix_str(f"Errors: {errros} || Success: {success} || Ratio: {success_ratio:.2f}%")
    
    # Group images by combinations of gender and ethnicity
    grouped_images = defaultdict(list)

    for image_annotations in annotations:
        gender = image_annotations["gender"]
        ethnicity = image_annotations["ethnicity"]
        grouped_images[(gender, ethnicity)].append(image_annotations)
    
    # Save images
    all_groups = list(sorted(grouped_images.keys()))
    pbar = tqdm(total=len(annotations), desc="Saving PATA images")
    ordered_annotations = []

    save_path = os.path.join(args.path, "images")
    os.makedirs(save_path, exist_ok=True)

    while any(grouped_images.values()):
        for group in all_groups:
            if grouped_images[group]:
                image = grouped_images[group].pop(0)
                gender = image["gender"]
                ethnicity = image["ethnicity"]
                age = image["age"]
                image_name = f"pata-{gender}-{ethnicity}-{age}-{len(ordered_annotations)}.png"
                img = Image.open(os.path.join(args.path, "images_tmp", image["name"])).convert("RGB")
                img.save(os.path.join(save_path, image_name), format="PNG")
            
                ordered_annotations.append({
                    "name": image_name,
                    "gender": gender,
                    "ethnicity": ethnicity,
                    "age": age,
                })

                pbar.update(1)
                pbar.refresh()

    pbar.close()
    ordered_annotations = pd.DataFrame(ordered_annotations)
    ordered_annotations.to_csv(os.path.join(args.path, "annotations.csv"), index=False)
    shutil.rmtree(os.path.join(args.path, "images_tmp"))
