import itertools
from typing import Any, Callable, Dict, Optional, Union, List

import spacy
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    EXAMPLE_DOC_STRING,
    rescale_noise_cfg
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteAttnProcessor
)
import numpy as np
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer


from compute_loss import get_attention_map_index_to_wordpiece, split_indices, calculate_positive_loss, calculate_negative_loss, get_indices, start_token, end_token, align_wordpieces_indices, extract_attribution_indices, extract_attribution_indices_with_verbs, extract_attribution_indices_with_verb_root, extract_entities_only




logger = logging.get_logger(__name__)



class SynGenDiffusionPipeline_add_relation(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 requires_safety_checker: bool = True,
                 include_entities: bool = False,
                 args=None,
                 ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)

        self.parser = spacy.load("en_core_web_trf")
        self.subtrees_indices = None
        self.doc = None
        self.include_entities = include_entities
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.args = args
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=vae_scale_factor)

    

    @staticmethod
    def _update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""
        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True
        )[0]
        latents = latents - step_size * grad_cond
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )
        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    def get_models(self):

        return self.vae, self.unet, self.scheduler, self.text_encoder, self.tokenizer, self.image_processor

   