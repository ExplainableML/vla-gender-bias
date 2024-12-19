import os
import torch
import pickle

from torch import Tensor
from peft import PeftModel
from .base import BaseVLM
from vlms.qwen import Qwen
from vlms.llava import LLaVA
from vlms.bunny import Bunny
from vlms.paligemma import PaliGemma
from vlms.internvl2 import InternVL2
from vlms.mobilevlm import MobileVLM
from vlms.llava_next import LLaVANext
from pruning_utils import apply_pruning
from pruning_utils import load_gradients
from pruning_utils import merge_gradients
from pruning_utils import get_importance_scores
from vlms.phi_3_vision_instruct import Phi3VModel
from utils.configs import model_to_variant_mapping


def patch_mobilevlm(model: BaseVLM, prompt_prefix: torch.nn.Parameter) -> None:
    original_prepare_multimodal = model.model.prepare_inputs_labels_for_multimodal

    def wrap_prepare_multimodal(self, *args, **kwargs):
        position_ids, attention_mask, past_key_values, new_input_embeds, new_labels = (
            original_prepare_multimodal(
                *args,
            )
        )

        prefix = (
            prompt_prefix.unsqueeze(0)
            .repeat(new_input_embeds.shape[0], 1, 1)
            .to(new_input_embeds.device, dtype=new_input_embeds.dtype)
        )
        new_input_embeds = torch.cat([new_input_embeds[:, :6, :], prefix, new_input_embeds[:, 6:, :]], dim=1)
        return (
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    model.model.prepare_inputs_labels_for_multimodal = wrap_prepare_multimodal.__get__(
        model.model, type(model.model)
    )


def patch_llava(model: BaseVLM, prompt_prefix: torch.nn.Parameter) -> None:
    original_prepare_multimodal = model.model.prepare_inputs_labels_for_multimodal

    def wrap_prepare_multimodal(self, *args, **kwargs):
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        ) = original_prepare_multimodal(
            *args,
        )

        prefix = (
            prompt_prefix.unsqueeze(0)
            .repeat(new_input_embeds.shape[0], 1, 1)
            .to(new_input_embeds.device, dtype=new_input_embeds.dtype)
        )
        new_input_embeds = torch.cat([new_input_embeds[:, :5, :], prefix, new_input_embeds[:, 5:, :]], dim=1)
        return (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    model.model.prepare_inputs_labels_for_multimodal = wrap_prepare_multimodal.__get__(
        model.model, type(model.model)
    )


def patch_internvl2(model: BaseVLM, prompt_prefix: torch.nn.Parameter) -> None:
    from vlms.internvl2 import InternVLPreprocessedPromptWithImage, IMG_CONTEXT_TOKEN

    def patched_get_next_token_probabilities(
        self, prompt: InternVLPreprocessedPromptWithImage
    ) -> Tensor:
        # Extract input_ids and image from prompts
        input_ids = prompt.input_ids.to(self.model.device)
        images = prompt.images.to(self.model.device, dtype=torch.bfloat16)
        attention_mask = prompt.attention_mask.to(self.model.device)

        # Pad input_ids and attention_mask
        input_ids_padding = torch.full(
            (input_ids.shape[0], prompt_prefix.shape[1]),
            fill_value=self.tokenizer.pad_token_id,
        )
        input_ids_padding = input_ids_padding.to(
            self.model.device, dtype=input_ids.dtype
        )
        input_ids = torch.cat([input_ids_padding, input_ids], dim=1)

        attention_mask_padding = torch.zeros(
            (attention_mask.shape[0], prompt_prefix.shape[1]),
        )
        attention_mask_padding = attention_mask_padding.to(
            self.model.device, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([attention_mask_padding, attention_mask], dim=1)

        # Find the input lengths
        input_lengths = attention_mask.sum(dim=1)
        # Find the start indices for the prompt prefix
        max_length = input_ids.shape[1]
        start_indices = max_length - input_lengths - prompt_prefix.shape[1]
        insert_indices = start_indices.unsqueeze(1).repeat(
            1, prompt_prefix.shape[0]
        ) + torch.arange(prompt_prefix.shape[0]).unsqueeze(0).to(start_indices.device)

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
        # Insert the prompt prefix using start_indices along the time dimension
        input_embeds[torch.arange(B).unsqueeze(1), insert_indices] = prompt_prefix.to(
            input_embeds.device, dtype=input_embeds.dtype
        )
        attention_mask[torch.arange(B).unsqueeze(1), insert_indices] = 1

        logits = self.model.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ).logits

        # Extract logits of last timestep and apply softmax
        next_token_logits = logits[:, -1, :]
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

        return next_token_probabilities

    model.get_next_token_probabilities = patched_get_next_token_probabilities.__get__(
        model, type(model)
    )


def prune_model(model: BaseVLM, model_name: str, sparsity: float) -> BaseVLM:
    path_to_importance_scores = os.path.join("./results/importance_scores", model_name)
    os.makedirs(path_to_importance_scores, exist_ok=True)
    path_to_importance_scores = os.path.join(
        path_to_importance_scores, "importance_scores.pkl"
    )

    if model_name.startswith("internvl2"):
        model_type = "internvl"
    else:
        model_type = "llama"

    try:
        with open(path_to_importance_scores, "rb") as ipf:
            importance_scores = pickle.load(ipf)
    except FileNotFoundError:
        # Load gradients
        gradients = load_gradients(
            model_name,
            only_gradients=False,
            bias_num_images=1000,
        )
        gradients = merge_gradients(gradients)

        # Calculate importance scores
        importance_scores = get_importance_scores(
            model, gradients, aggregation_method="sum", model_type=model_type
        )
        with open(path_to_importance_scores, "wb") as ipf:
            pickle.dump(importance_scores, ipf, fix_imports=True)
        
    apply_pruning(
        model=model,
        sparsity=sparsity,
        granularity="local",
        importance_scores=importance_scores,
        model_type=model_type,
    )

    return model


def load_model(name: str) -> BaseVLM:
    # Naming convention for tuned models: tuned:MODELNAME:TASK:TUNINGMODE
    if name.startswith("tuned:"):
        _, name, task, tuning_mode = name.split(":")
        path_to_tuned_model = os.path.join("./results/tuned_models", name, task)
        model = load_model(name)
        if tuning_mode == "full":
            state_dict = torch.load(
                os.path.join(path_to_tuned_model, "full", "model.pt"),
                map_location="cpu",
            )
            model.model.load_state_dict(state_dict)
        elif tuning_mode == "lora":
            llm_layers = model.get_llm_layers()
            llm_layers_lora = PeftModel.from_pretrained(
                llm_layers,
                os.path.join(path_to_tuned_model, "lora"),
            )
            llm_layers_lora.merge_and_unload()
        else:
            raise ValueError(f"Unknown tuning mode {tuning_mode}")

        return model
    
    if name.startswith("pruned"):
        _, name, sparsity = name.split(":")
        model = load_model(name)
        model = prune_model(model, name, float(sparsity))
        return model
    
    if name.startswith("prompt-tuned"):
        _, name = name.split(":")
        model = load_model(name)
        prefix = torch.load(os.path.join("./results/prompt_tuning", name, "tuning_prefix.pt"))

        if name == "internvl2-8b":
            patch_internvl2(model, prefix)
        elif name.startswith("llava"):
            patch_llava(model, prefix)
        elif name.startswith("mobilevlm"):
            patch_mobilevlm(model, prefix)
        else:
            raise ValueError(f"Unknown model {name}")
        
        return model

    if name in model_to_variant_mapping:
        variant = model_to_variant_mapping[name]
        if name == "qwen":
            return Qwen(variant=variant)
        elif name.startswith("mobilevlm"):
            return MobileVLM(variant=variant)
        elif name == "bakllava":
            return LLaVA(variant=variant)
        elif name.startswith("llava") and "1.6" in variant:
            return LLaVANext(variant=variant)
        elif name.startswith("llava"):
            return LLaVA(variant=variant)
        elif name.startswith("bunny"):
            return Bunny(variant=variant)
        elif name.startswith("internvl2"):
            return InternVL2(variant=variant)
        elif name.startswith("phi"):
            return Phi3VModel(variant=variant)
        elif name.startswith("paligemma"):
            return PaliGemma(variant=variant)
    else:
        raise ValueError(f"Unknown model {name}")
