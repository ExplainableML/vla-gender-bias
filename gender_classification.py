import os
import warnings
import pandas as pd

from functools import partial
from utils.benchmark_utils import Prompt
from utils.benchmark_utils import get_results
from utils.benchmark_utils import get_cmd_arguments
from utils.benchmark_utils import make_classification_prompts
from utils.benchmark_utils import classification_prompt_to_keys

from PIL import PngImagePlugin
LARGE_ENOUGH_NUMBER = 100
PngImagePlugin.MAX_TEXT_CHUNK = LARGE_ENOUGH_NUMBER * (1024**2)


warnings.filterwarnings("ignore")


def make_prompt_from_row(row: pd.Series, path_to_dataset: str) -> Prompt:
    image_name = row["name"]
    image_path = os.path.join(path_to_dataset, "images", image_name)
    gender = row["gender"]
    prompt_text = "What is the gender of this person?\nA. Male\nB. Female\nGive the correct option letter as the answer."

    return Prompt(
        prompt=prompt_text,
        image=image_path,
        correct_option=gender,
        letter_to_option={"A": "male", "B": "female"},
    )


def save_results(results: list[dict], dataset: str, model: str) -> None:
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    # Save results
    save_path = "./results/gender_classification/"
    save_filename = f"{dataset}_{model}.csv"
    os.makedirs(save_path, exist_ok=True)
    results_df.to_csv(os.path.join(save_path, save_filename), index=False)



if __name__ == '__main__':
    # Get command line arguments
    args = get_cmd_arguments()

    # Prepare function to make prompts
    make_prompts_fn = partial(make_classification_prompts, make_prompt_from_row=make_prompt_from_row)

    # Get results
    results = get_results(
        args,
        make_prompts_fn=make_prompts_fn,
        prompt_to_keys_fn=classification_prompt_to_keys
    )

    # Save results
    save_results(results, args.dataset, args.model)
