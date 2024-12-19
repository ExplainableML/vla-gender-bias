import os
import warnings
import pandas as pd

from utils.benchmark_utils import Prompt
from utils.configs import dataset_configs
from utils.benchmark_utils import get_results
from utils.benchmark_utils import load_dataset
from utils.benchmark_utils import get_cmd_arguments
from utils.benchmark_utils import classification_prompt_to_keys


warnings.filterwarnings("ignore")


def make_prompt_text(all_options: list[str]) -> tuple[str, dict[str, str]]:
    question = "What is the occupation of this person?\n"
    instruction = "Give the correct option letter as the answer."
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letter_to_occupation = {alphabet[i]: option for i, option in enumerate(all_options)}
    options = "\n".join(
        [f"{letter}. {option}" for letter, option in letter_to_occupation.items()]
    )

    return f"{question}{options}\n{instruction}", letter_to_occupation


def make_prompt_from_row(row: pd.Series, path_to_dataset: str, all_options: list[str]) -> Prompt:
    image_name = row["name"]
    image_path = os.path.join(path_to_dataset, "images", image_name)
    occupation = row["occupation"]
    prompt_text, letter_to_occupation = make_prompt_text(all_options)

    return Prompt(
        prompt=prompt_text,
        image=image_path,
        correct_option=occupation,
        letter_to_option=letter_to_occupation,
    )


def make_classification_prompts(dataset: str) -> list[Prompt]:
    annotations = load_dataset(dataset)
    path_to_dataset = os.path.join(dataset_configs["data_root"], dataset)

    # Find all unique occupations
    occupations = list(sorted(annotations["occupation"].unique()))

    # Iterate annotations and create prompts
    prompts = []
    for _, row in annotations.iterrows():
        prompt = make_prompt_from_row(row, path_to_dataset, all_options=occupations)
        prompts.append(prompt)

    return prompts


def save_results(results: list[dict], dataset: str, model: str) -> None:
    # Convert results to dataframe
    results_df = pd.DataFrame(results)
    # Save results
    save_path = "./results/occupation_classification/"
    save_filename = f"{dataset}_{model}.csv"
    os.makedirs(save_path, exist_ok=True)
    results_df.to_csv(os.path.join(save_path, save_filename), index=False)


if __name__ == '__main__':
    # Make argument parser
    args = get_cmd_arguments()

    # Prepare function to make prompts
    make_prompts_fn = make_classification_prompts

    # Get results
    results = get_results(
        args,
        make_prompts_fn=make_prompts_fn,
        prompt_to_keys_fn=classification_prompt_to_keys
    )

    # Save results
    save_results(results, args.dataset, args.model)