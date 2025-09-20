"""ComfyUI custom node for sd_embed long prompt weighting.

This module exposes a single node that can be dropped into ComfyUI's
``custom_nodes`` folder.  The node re-uses the existing long prompt weighting
helpers from :mod:`sd_embed.embedding_funcs` to provide unlimited length
prompt/negative prompt encoding directly inside ComfyUI while remaining
compatible with the standard ``CheckpointLoader``/``CLIPLoader`` ->
``KSampler`` workflow.

The implementation focuses on Stable Diffusion 1.5 style checkpoints as well
as SDXL checkpoints.  Model type detection happens automatically, but it can
also be forced through the node's ``model_type`` drop-down.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

from .embedding_funcs import (
    get_weighted_text_embeddings_sd15,
    get_weighted_text_embeddings_sdxl,
)


@dataclass
class _ResolvedClipComponents:
    """Container holding the objects required by the embedding helpers."""

    tokenizer: Any
    text_encoder: Any
    device: torch.device
    tokenizer_2: Optional[Any] = None
    text_encoder_2: Optional[Any] = None
    text_encoder_3: Optional[Any] = None

    def as_sd15_pipe(self) -> SimpleNamespace:
        return SimpleNamespace(
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            device=self.device,
        )

    def as_sdxl_pipe(self) -> SimpleNamespace:
        return SimpleNamespace(
            tokenizer=self.tokenizer,
            tokenizer_2=self.tokenizer_2,
            text_encoder=self.text_encoder,
            text_encoder_2=self.text_encoder_2,
            text_encoder_3=self.text_encoder_3,
            device=self.device,
        )


def _first_attr(obj: Any, paths: Iterable[str]) -> Any:
    """Return the first non-``None`` attribute found following *paths*.

    Each path may include ``.`` separated attribute names allowing us to look
    through nested objects (e.g. ``"clip_l.transformer"``).
    """

    for path in paths:
        current = obj
        try:
            for part in path.split("."):
                if current is None:
                    break
                current = getattr(current, part)
            if current is not None:
                return current
        except AttributeError:
            continue
    return None


def _module_device(module: Any) -> Optional[torch.device]:
    if module is None:
        return None
    device = getattr(module, "device", None)
    if device is not None:
        return torch.device(device)
    if hasattr(module, "parameters"):
        try:
            return next(module.parameters()).device
        except (StopIteration, AttributeError):
            return None
    return None


def _resolve_clip_components(clip: Any) -> _ResolvedClipComponents:
    """Extract tokenizer(s) and text encoder(s) from a ComfyUI ``clip`` object."""

    cond_stage = getattr(clip, "cond_stage_model", clip)

    tokenizer = _first_attr(
        cond_stage,
        (
            "tokenizer",
            "tokenizer1",
            "tokenizer_1",
            "clip_l.tokenizer",
            "clip.tokenizer",
        ),
    )
    text_encoder = _first_attr(
        cond_stage,
        (
            "text_encoder",
            "transformer",
            "clip_l.transformer",
            "clip.text_encoder",
            "clip.transformer",
        ),
    )

    tokenizer_2 = _first_attr(
        cond_stage,
        (
            "tokenizer_2",
            "tokenizer2",
            "clip_g.tokenizer",
            "clip2.tokenizer",
        ),
    )
    text_encoder_2 = _first_attr(
        cond_stage,
        (
            "text_encoder_2",
            "transformer2",
            "clip_g.transformer",
            "clip2.transformer",
        ),
    )
    text_encoder_3 = _first_attr(
        cond_stage,
        (
            "text_encoder_3",
            "transformer3",
        ),
    )

    if tokenizer is None or text_encoder is None:
        raise RuntimeError(
            "Unable to resolve tokenizer/text encoder from the provided CLIP object."
        )

    device = (
        _module_device(text_encoder)
        or _module_device(text_encoder_2)
        or getattr(cond_stage, "device", None)
        or getattr(clip, "device", None)
        or torch.device("cpu")
    )

    return _ResolvedClipComponents(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        tokenizer_2=tokenizer_2,
        text_encoder_2=text_encoder_2,
        text_encoder_3=text_encoder_3,
        device=torch.device(device),
    )


def _detect_model_type(components: _ResolvedClipComponents) -> str:
    if components.tokenizer_2 is not None and components.text_encoder_2 is not None:
        return "sdxl"
    return "sd15"


class SDLongPromptWeightedEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "negative_text": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
            },
            "optional": {
                "model_type": (
                    ["auto", "sd15", "sdxl"],
                    {"default": "auto"},
                ),
                "clip_skip": (
                    "INT",
                    {"default": 0, "min": 0, "max": 12},
                ),
                "pad_last_block": (
                    "BOOLEAN",
                    {"default": False},
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "encode"
    CATEGORY = "conditioning"

    def encode(
        self,
        clip: Any,
        text: str,
        negative_text: str,
        model_type: str = "auto",
        clip_skip: int = 0,
        pad_last_block: bool = False,
    ) -> Tuple[Any, Any]:
        components = _resolve_clip_components(clip)
        detected_type = _detect_model_type(components)
        variant = model_type if model_type != "auto" else detected_type

        positive_base = clip.encode(text if text else "")
        negative_base = clip.encode(negative_text if negative_text else "")
        reference_positive = positive_base[0][0]
        reference_negative = negative_base[0][0]

        if variant == "sdxl":
            if components.tokenizer_2 is None or components.text_encoder_2 is None:
                raise RuntimeError(
                    "The provided CLIP object does not expose the secondary tokenizer/"
                    "encoder required for SDXL embeddings."
                )

            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = get_weighted_text_embeddings_sdxl(
                pipe=components.as_sdxl_pipe(),
                prompt=text,
                neg_prompt=negative_text,
                pad_last_block=pad_last_block,
                clip_skip=clip_skip or None,
            )

            if reference_positive is not None:
                prompt_embeds = prompt_embeds.to(
                    device=reference_positive.device,
                    dtype=reference_positive.dtype,
                )
            if reference_negative is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=reference_negative.device,
                    dtype=reference_negative.dtype,
                )

            positive_base[0][0] = prompt_embeds
            negative_base[0][0] = negative_prompt_embeds

            if pooled_prompt_embeds is not None:
                positive_base[0][1]["pooled_output"] = pooled_prompt_embeds
            if negative_pooled_prompt_embeds is not None:
                negative_base[0][1]["pooled_output"] = negative_pooled_prompt_embeds

        else:
            prompt_embeds, negative_prompt_embeds = get_weighted_text_embeddings_sd15(
                pipe=components.as_sd15_pipe(),
                prompt=text,
                neg_prompt=negative_text,
                pad_last_block=pad_last_block,
                clip_skip=clip_skip,
            )

            if reference_positive is not None:
                prompt_embeds = prompt_embeds.to(
                    device=reference_positive.device,
                    dtype=reference_positive.dtype,
                )
            if reference_negative is not None:
                negative_prompt_embeds = negative_prompt_embeds.to(
                    device=reference_negative.device,
                    dtype=reference_negative.dtype,
                )

            positive_base[0][0] = prompt_embeds
            negative_base[0][0] = negative_prompt_embeds

        return positive_base, negative_base


NODE_CLASS_MAPPINGS: Dict[str, Any] = {
    "SDLongPromptWeightedEncode": SDLongPromptWeightedEncode,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "SDLongPromptWeightedEncode": "SD Long Prompt Weighted Encode",
}

