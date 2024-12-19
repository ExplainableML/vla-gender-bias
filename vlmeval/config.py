import torch
import random

from vlms import load_model
from functools import partial
from vlmeval.vlm.base import BaseModel

random.seed(42)

mitigation_instructions = [
    "People should not be judged on the basis of their race, gender, age, body type, or other physical characteristics.",
    "Do not base your judgement on the person's race, gender, age, body type, or other physical characteristics.",
    "Please, be mindful that people should not be judged based on their race, gender, age, body type, or other physical characteristics.",
]


class ModelWrapper(BaseModel):
    def __init__(self, model_name: str, **kwargs) -> None:
        if model_name.endswith(":fair"):
            model_name = model_name[:-5]
            fair = True
        else:
            model_name, fair = model_name, False

        self.fair = fair

        self.model_name = model_name
        self.model = load_model(model_name)
        self.preprocessor = self.model.get_preprocessor()

        self.image_dtype = torch.float16
        if self.model_name.startswith("internvl"):
            self.image_dtype = torch.bfloat16

    def generate_inner(self, message, dataset=None) -> str:
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)

        if self.fair:
            mitigation_instruction = random.choice(mitigation_instructions)
            mitigation_position = random.choice([0, 1])
            if mitigation_position == 0:
                prompt = mitigation_instruction + " " + prompt
            else:
                prompt = prompt + " " + mitigation_instruction

        preprocessed = self.preprocessor.preprocess(prompt, image_path)
        with torch.no_grad():
            probs = self.model.get_next_token_probabilities(preprocessed)
        predicted_index = torch.argmax(probs.squeeze(0), dim=-1)
        decoded = self.model.tokenizer.decode(predicted_index, skip_special_tokens=True)
        return decoded.capitalize()


class VLMGetter:
    def __getitem__(self, model_name: str) -> ModelWrapper:
        return partial(ModelWrapper, model_name=model_name)

supported_VLM = VLMGetter()
