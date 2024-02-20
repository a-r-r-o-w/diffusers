from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from ...image_processor import PipelineImageInput
from .pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline


class SDXLDifferentialDiffusionPipeline(StableDiffusionXLImg2ImgPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        prompt_2: Optional[Union[str, List[str]]] = None,
        image: PipelineImageInput = None,
        strength: float = 0.3,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        denoising_start: Optional[float] = None,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
        negative_original_size: Optional[Tuple[int, int]] = None,
        negative_crops_coords_top_left: Tuple[int, int] = (0, 0),
        negative_target_size: Optional[Tuple[int, int]] = None,
        aesthetic_score: float = 6.0,
        negative_aesthetic_score: float = 2.5,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        # Differential Diffusion specific
        map: torch.FloatTensor = None,
        **kwargs,
    ):
        if map is None:
            raise ValueError("`map` must be provided as a parameter")

        original_with_noise = thresholds = masks = None

        def callback_on_step_begin(i, t, callback_kwargs):
            nonlocal original_with_noise, thresholds, masks

            timesteps = callback_kwargs.get("timesteps")
            batch_size = callback_kwargs.get("batch_size")
            prompt_embeds = callback_kwargs.get("prompt_embeds")
            device = callback_kwargs.get("device")
            latents = callback_kwargs.get("latents")

            if i == 0:
                original_with_noise = self.prepare_latents(
                    image,
                    timesteps,
                    batch_size,
                    num_images_per_prompt,
                    prompt_embeds.dtype,
                    device,
                    generator,
                )
                thresholds = torch.arange(num_inference_steps, dtype=map.dtype) / num_inference_steps
                thresholds = thresholds.unsqueeze(1).unsqueeze(1).to(device)
                masks = map > (thresholds + (denoising_start or 0))

                if denoising_start is None:
                    latents = original_with_noise[:1]
            else:
                mask = masks[i].unsqueeze(0)
                mask = mask.to(latents.dtype)
                mask = mask.unsqueeze(1)
                latents = original_with_noise[i] * mask + latents * (1 - mask)

            return latents

        callback_on_step_begin_tensor_inputs = ["timesteps", "batch_size", "prompt_embeds", "device", "latents"]

        return super().__call__(
            prompt=prompt,
            prompt_2=prompt_2,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            denoising_start=denoising_start,
            denoising_end=denoising_end,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_image_embeds=ip_adapter_image_embeds,
            output_type=output_type,
            return_dict=return_dict,
            cross_attention_kwargs=cross_attention_kwargs,
            guidance_rescale=guidance_rescale,
            original_size=original_size,
            crops_coords_top_left=crops_coords_top_left,
            target_size=target_size,
            negative_original_size=negative_original_size,
            negative_crops_coords_top_left=negative_crops_coords_top_left,
            negative_target_size=negative_target_size,
            aesthetic_score=aesthetic_score,
            negative_aesthetic_score=negative_aesthetic_score,
            clip_skip=clip_skip,
            callback_on_step_begin=callback_on_step_begin,
            callback_on_step_begin_tensor_inputs=callback_on_step_begin_tensor_inputs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            **kwargs,
        )
