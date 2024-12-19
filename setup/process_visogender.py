import os
import argparse
import requests
import pandas as pd
import urllib.request as urlopen

from tqdm import tqdm
from PIL import Image
from urllib.error import HTTPError
from PIL import UnidentifiedImageError


def save_image(url, target_path):
    try:
        img = Image.open(requests.get(url, stream=True).raw)
    except UnidentifiedImageError:
        opener = urlopen.build_opener()
        opener.addheaders = [('User-Agent', 'Chrome')]
        urlopen.install_opener(opener)
        try:
            urlopen.urlretrieve(url, target_path)
        except HTTPError:
            return
        except UnidentifiedImageError:
            return
                    
        try:
            img = Image.open(target_path)
        except UnidentifiedImageError:
            # print(f"UnidentifiedImageError {idx}: {url}")
            os.remove(target_path)
            return
    except:
        return

    img.save(target_path, format="png")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()

    files = os.listdir(args.path)
    assert len(files) == 2
    assert "OO_Visogender_02102023.tsv" in files
    assert "OP_Visogender_11012024.tsv" in files
    
    annotations = []
    pbar = tqdm(total=680, desc="Downloading Visogender")
    os.makedirs(os.path.join(args.path, "images"), exist_ok=True)

    gender_mapping = {
        "masculine": "male",
        "feminine": "female",
    }
    
    for file in files:
        file_annotations = pd.read_csv(os.path.join(args.path, file), sep="\t")
        for _, row in file_annotations.iterrows():
            url = row["URL type (Type NA if can't find)"]
            idx = row["IDX"]
            occupation = row["Occupation"]
            gender = row["Occupation_perceived_gender"]
            gender = gender_mapping[gender]

            name = f"{occupation.lower()}_{gender.lower()}_{idx}.png"
            target_path = os.path.join(args.path, "images", name)
            if os.path.exists(target_path):
                continue

            status = save_image(url, target_path)
            if status is None:
                continue

            annotations.append(
                {
                    "name": name,
                    "occupation": occupation,
                    "gender": gender,
                    "id": idx,
                }
            )
            pbar.update(1)
    
    pbar.close()
    
    annotations = pd.DataFrame(annotations)
    annotations.to_csv(os.path.join(args.path, "annotations.csv"), index=False)
