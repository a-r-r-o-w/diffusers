import gc

import torch
from diffusers import LattePipeline
from diffusers.utils import export_to_gif
from transformers import T5EncoderModel, BitsAndBytesConfig


def flush():
    gc.collect()
    torch.cuda.empty_cache()

def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


model_id = "maxin-cn/Latte-1"
text_encoder = T5EncoderModel.from_pretrained(
    model_id,
    subfolder="text_encoder",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
    device_map="auto",
)
pipe = LattePipeline.from_pretrained(
    model_id, 
    text_encoder=text_encoder,
    transformer=None,
    vae=None,
    device_map="balanced",
)

with torch.no_grad():
    prompt = "A dog in astronaut suit and sunglasses floating in space"
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(prompt)

del text_encoder
del pipe
flush()

pipe = LattePipeline.from_pretrained(
    model_id,
    text_encoder=None,
    torch_dtype=torch.float16,
).to("cuda")
# pipe.enable_vae_tiling()
# pipe.enable_vae_slicing()

video = pipe(
    video_length=16,
    num_inference_steps=25,
    negative_prompt=None, 
    prompt_embeds=prompt_embeds,
    negative_prompt_embeds=negative_prompt_embeds,
).frames[0]

print(
    f"Max memory allocated: {bytes_to_giga_bytes(torch.cuda.max_memory_allocated())} GB"
)
export_to_gif(video, "latte_memory_optimized.gif")
