import torch

from math import floor
from torch import Tensor
from .base import BaseVLM
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, Optional
from .base import BasePreprocessor
from utils.configs import model_configs
from accelerate import init_empty_weights
from accelerate import infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch
from vlms.backbones.internvl2.utils import load_image
from .backbones.internvl2.conversation import get_conv_template
from .base import PreprocessedPrompt, PreprocessedPromptWithImage
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer as InternVL1BTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer as InternVL4BTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer as InternVL40BTokenizer
from .backbones.internvl2.internvl2_1b.modeling_internvl_chat import InternVLChatModel as InternVL1B
from .backbones.internvl2.internvl2_2b.modeling_internvl_chat import InternVLChatModel as InternVL2B
from .backbones.internvl2.internvl2_4b.modeling_internvl_chat import InternVLChatModel as InternVL4B
from .backbones.internvl2.internvl2_8b.modeling_internvl_chat import InternVLChatModel as InternVL8B
from .backbones.internvl2.internvl2_26b.modeling_internvl_chat import InternVLChatModel as InternVL26B
from .backbones.internvl2.internvl2_40b.modeling_internvl_chat import InternVLChatModel as InternVL40B
from .backbones.internvl2.internvl2_2b.tokenization_internlm2 import InternLM2Tokenizer as InternVL2BTokenizer
from .backbones.internvl2.internvl2_8b.tokenization_internlm2 import InternLM2Tokenizer as InternVL8BTokenizer
from .backbones.internvl2.internvl2_26b.tokenization_internlm2 import InternLM2Tokenizer as InternVL26BTokenizer


IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


variant_to_cls_mapping = {
    "internvl2-1b": (InternVL1B, InternVL1BTokenizer),
    "internvl2-2b": (InternVL2B, InternVL2BTokenizer),
    "internvl2-4b": (InternVL4B, InternVL4BTokenizer),
    "internvl2-8b": (InternVL8B, InternVL8BTokenizer),
    "internvl2-26b": (InternVL26B, InternVL26BTokenizer),
    "internvl2-40b": (InternVL40B, InternVL40BTokenizer),
}


variant_to_memory_mapping = {
    "internvl2-1b": 3,
    "internvl2-2b": 6,
    "internvl2-4b": 9,
    "internvl2-8b": 17,
    "internvl2-26b": 55,
    "internvl2-40b": 80,
}


@dataclass
class InternVLPreprocessedPrompt(PreprocessedPrompt):
    attention_mask: Tensor


@dataclass
class InternVLPreprocessedPromptWithImage(PreprocessedPromptWithImage):
    attention_mask: Optional[Tensor] = None
    num_patches_list: Optional[list[int]] = None


class InternVL2Preprocessor(BasePreprocessor):
    def __init__(self, tokenizer, img_context_token_id: int, template, system_message, num_image_token) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.img_context_token_id = img_context_token_id
        self.template = template
        self.system_message = system_message
        self.num_image_token = num_image_token

    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> InternVLPreprocessedPromptWithImage:
        # Make prompts and images lists
        prompts = [prompts] if isinstance(prompts, str) else prompts
        images = [images] if isinstance(images, str) else images

        # Load images
        images = [load_image(image, max_num=12) for image in images]

        # Make `num_patches_list`
        num_patches_list = [image.size(0) for image in images]

        # Concatenate images
        images = torch.cat(images, dim=0)

        # Process prompts
        processed_prompts = []
        for idx, num_patches in enumerate(num_patches_list):
            prompt = prompts[idx]
            if "<image>" not in prompt:
                prompt = "<image>\n" + prompt

            template = get_conv_template(self.template)
            template = deepcopy(template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            prompt = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            prompt = prompt.replace("<image>", image_tokens, 1)
            processed_prompts.append(prompt)

        # Tokenize prompts
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(
            processed_prompts, return_tensors="pt", padding=True
        )
        input_ids = model_inputs["input_ids"]

        # Return
        return InternVLPreprocessedPromptWithImage(
            input_ids=input_ids,
            input_lengths=None,
            images=images,
            attention_mask=model_inputs.attention_mask,
            num_patches_list=num_patches_list,
        )

    def preprocess_for_lm(
        self, prompts: Union[str, list[str]]
    ) -> InternVLPreprocessedPrompt:
        # Make sure prompts is a list
        prompts = [prompts] if isinstance(prompts, str) else prompts

        # Process prompts
        processed_prompts = []
        for prompt in prompts:
            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            prompt = template.get_prompt()
            processed_prompts.append(prompt)

        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(
            processed_prompts, return_tensors="pt", padding=True
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        return InternVLPreprocessedPrompt(
            input_ids=input_ids,
            input_lengths=None,
            attention_mask=attention_mask,
        )


class InternVL2(BaseVLM):
    def __init__(self, variant: str):
        # Get path to model
        model_path = model_configs["internvl2"][variant]["model_path"]

        # Get model class
        model_cls, tokenizer_cls = variant_to_cls_mapping[variant]

        # Load model
        with init_empty_weights():
            model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
            )

            # Determine how much memory to use on each device
            num_gpus = torch.cuda.device_count()
            required_memory = variant_to_memory_mapping[variant]
            memory_per_gpu = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(num_gpus)
            ]
            memory_per_gpu = [floor(memory / 1024**3) for memory in memory_per_gpu]
            total_available_memory = sum(memory_per_gpu)
            memory_per_gpu = [
                min(memory / total_available_memory * required_memory, memory)  # Can't use more memory than available
                for memory in memory_per_gpu
            ]
            max_memory = {i: f"{memory}GiB" for i, memory in enumerate(memory_per_gpu)}
            max_memory["cpu"] = "0GiB"

            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[
                    "LlamaDecoderLayer",
                    "InternVisionEncoderLayer",
                    "InternLM2DecoderLayer",
                ],
            )

        model = load_checkpoint_and_dispatch(
            model,
            model_path,
            device_map=device_map,
        )
        self.model = model.eval()

        self.tokenizer = tokenizer_cls.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

    def get_next_token_probabilities(self, prompt: InternVLPreprocessedPromptWithImage) -> Tensor:
        # Extract input_ids and image from prompts
        input_ids = prompt.input_ids.to(self.model.device)
        images = prompt.images.to(self.model.device, dtype=torch.bfloat16)
        attention_mask = prompt.attention_mask.to(self.model.device)

        # Set img_context_token_id
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        # Forward pass
        vit_embeds = self.model.extract_feature(images)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = torch.eq(input_ids, self.model.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
        logits = self.model.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ).logits

        # Extract logits of last timestep and apply softmax
        next_token_logits = logits[:, -1, :]
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

        return next_token_probabilities

    def get_next_token_probabilities_from_lm(self, prompt: InternVLPreprocessedPrompt) -> Tensor:
        # Extract input_ids and image from prompts
        input_ids = prompt.input_ids.to(self.model.device)
        attention_mask = prompt.attention_mask.to(self.model.device)

        # Forward pass
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        logits = self.model.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ).logits

        # Extract logits of last timestep and apply softmax
        next_token_logits = logits[:, -1, :]
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

        return next_token_probabilities

    def get_preprocessor(self) -> InternVL2Preprocessor:
        return InternVL2Preprocessor(
            tokenizer=self.tokenizer,
            img_context_token_id=self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
            template=self.model.template,
            system_message=self.model.system_message,
            num_image_token=self.model.num_image_token,
        )

    def get_llm_layers(self) -> torch.nn.ModuleList:
        return self.model.language_model.model.layers

    def get_embedding_size(self) -> int:
        return self.model.language_model.model.tok_embeddings.embedding_dim
    
    def get_avg_embedding(self) -> Tensor:
        return self.model.language_model.model.tok_embeddings.weight.mean(dim=0)
