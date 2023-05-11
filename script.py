# Code adpated from "Prompt-to-Prompt Image Editing with Cross Attention Control"
# https://github.com/bloc97/CrossAttentionControl

import random
from functools import partial
import einops
import numpy as np
import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from typing import List


# Overwrites CrossAttention._sliced_attention and CrossAttention._attention for all modules of the u-net
def init_attention_func(unet, vae, clip, device, clip_tokenizer):
    """Drop-in replacement for CrossAttention._attention, which ignores the key and value arguments and instead uses hardcoded
    prompt embeddings from self.mapping.
    Returns the hidden states of the attention layer, i.e. it returns a convex combination over projected prompt embeddings
    (with the convex coefficients being the attention probabilities)."""

    def new_attention(self: "CrossAttention", query, key, value):
        del (
            key,
            value,
        )  # We ignore these arguments! Instead, we accept them as hardcoded instance variables in `self.mapping`.
        ## query shape: (B*H, n_pixels, dim_head)
        ## key.T shape: (B*H, dim_head, T)

        n_pixels = query.shape[1]
        W = H = int(n_pixels**0.5)
        assert W**2 == n_pixels
        hidden_states = 0.0

        # We create self.mappings as a List[Tuple[torch.Tensor, Callable[[int, int], torch.Tensor]]]
        for prompt_emb, mask_fn in self.mappings:
            mask = mask_fn(W, H)
            mask = mask.reshape(1, n_pixels, 1)
            mask = mask.to(device)

            # reshape_heads: (B, H, n_pixels, dim_head) => (B*H, n_pixels, dim_head)
            k = self.reshape_heads_to_batch_dim(
                self.to_k(prompt_emb)
            )  # Project the prompt embedding to the key space (including multiple heads).
            v = self.reshape_heads_to_batch_dim(
                self.to_v(prompt_emb)
            )  # Project the prompt embedding to the key space (including multiple heads).

            scores = (
                torch.matmul(query, k.transpose(-1, -2)) * self.scale
            )  ## shape: (batch*n_heads, n_pixels, seq_len)
            # mask = einops.repeat(mask, "1 P 1 -> B P T", B=scores.shape[0], T=scores.shape[2])  # Expand indices for indexing two lines down.
            # scores[~mask] = -float("inf")
            # attn_probs[~mask] = 0  # Replace nans with 0s.

            # softmax over the prompt tokens
            attn_probs = scores.softmax(dim=-1)
            # Later: try attn_probs
            attn_probs *= mask

            hidden_states += torch.matmul(
                attn_probs, v
            )  # Reshape attn_slice (BH, 1, T)  => (BH, T, 1)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        # [8, 4096, 40]
        return hidden_states

    def new_sliced_attention(self, query, key, value, sequence_length, dim):
        # We don't implement this case (slicing optimized case) but want to error out when it is called.
        raise NotImplementedError

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if (
            module_name == "CrossAttention" and "attn2" in name
        ):  # "attn2" is cross-attention. "attn1" is (equivalent to) self-attention.
            module.mapping = None
            module._sliced_attention = new_sliced_attention.__get__(
                module, type(module)
            )
            module._attention = new_attention.__get__(module, type(module))

def _get_prompt_emb(prompt, clip, clip_tokenizer, device):
    tokens_conditional = clip_tokenizer(
        prompt,
        padding="max_length",
        max_length=clip_tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
        return_overflowing_tokens=True,
    )
    embedding_conditional = clip(
        tokens_conditional.input_ids.to(device)
    ).last_hidden_state
    return embedding_conditional


def left_right_exp(
    left_prompt,
    right_prompt,
    *, 
    unet,
    vae,
    clip,
    device,
    clip_tokenizer, transpose=False, batch_size=1, **kwargs):
    kwargs["batch_size"] = batch_size
    def get_prompt_emb(prompt):
        emb = _get_prompt_emb(prompt, clip=clip, clip_tokenizer=clip_tokenizer, device=device)
        return emb.expand(batch_size, -1, -1)

    uncond_emb = get_prompt_emb("")
    left_emb = get_prompt_emb(left_prompt)
    right_emb = get_prompt_emb(right_prompt)
    neg_embed_u = get_prompt_emb(left_prompt + right_prompt)
    neg_embed_c = -get_prompt_emb(left_prompt + right_prompt)
    dummy_emb = torch.zeros_like(uncond_emb).to(device)

    # If transpose=True, wrap each tensor in a wrap_transpose() call.
    # This changes left-right relationship to top-bottom relationship.
    T = wrap_transpose if transpose else lambda x: x

    # Assign the paired prompt embeddings and mask-making functions to each cross-attention layer.
    # The new_attention() function will read these embeddings and masks and ignore its encoder_hidden_state argument.
    def use_unconditional_mappings():
        # Mapping associated with unconditional denoising.
        for name, module in unet.named_modules():
            if type(module).__name__ == "CrossAttention" and "attn2" in name:
                #module.mappings = ((uncond_emb, partial(make_centre_vertical_mask, percent=0.2)),)
                module.mappings = ((neg_embed_u, T(partial(make_centre_vertical_mask, percent=0.2))),
                                    (uncond_emb, T(partial(make_left_mask, percent=0.8))),
                                    (uncond_emb, T(partial(make_right_mask, percent=0.8))),)
            else:
                module.mappings = None

    def use_conditional_mappings():
        # Mapping associated with conditional denoising (includes separate left and right embeddings).
        for name, module in unet.named_modules():
            if type(module).__name__ == "CrossAttention" and "attn2" in name:
                module.mappings = (
                    (left_emb, T(partial(make_left_mask, percent=0.8))),
                    (right_emb, T(partial(make_right_mask, percent=0.8))),
                    (uncond_emb, T(partial(make_centre_vertical_mask, percent=0.2))),
                )
            else:
                module.mappings = None

    return stablediffusion(use_unconditional_mappings, use_conditional_mappings,
                            unet=unet, vae=vae, clip=clip, device=device, clip_tokenizer=clip_tokenizer, **kwargs)

def top_bottom_exp(top_prompt, bottom_prompt, **kwargs):
    return left_right_exp(top_prompt, bottom_prompt, transpose=True, **kwargs)
                           

def grid_exp(
    grid_array: list,
    *, 
    unet,
    vae,
    clip,
    device,
    clip_tokenizer, batch_size=1, **kwargs,
) -> List[Image]:
    kwargs["batch_size"] = batch_size
    def get_prompt_emb(prompt):
        emb = _get_prompt_emb(prompt, clip=clip, clip_tokenizer=clip_tokenizer, device=device)
        return emb.expand(batch_size, -1, -1)

    uncond_emb = get_prompt_emb("")

    def use_grid_mappings_cond():
        mappings = []
        keys = set()
        for row in grid_array:
            for k in row:
                    keys.add(k)
        for key in keys:
            if key is None:
                emb = uncond_emb
            else:
                emb = get_prompt_emb(key)
            mappings.append((emb, partial(make_grid_mask, key=key, grid_array=grid_array)))

        for name, module in unet.named_modules():
            if type(module).__name__ == "CrossAttention" and "attn2" in name:
                module.mappings = mappings
            else:
                module.mappings = None
        return mappings

    def use_grid_mappings_uncond():
        mappings = []
        keys = set()
        for row in grid_array:
            for k in row:
                keys.add(k)
        for key in keys:
            if key is not None:
                emb = uncond_emb
            else:
                emb = get_prompt_emb(" ".join([k for k in keys if k is not None]))
            mappings.append((emb, partial(make_grid_mask, key=key, grid_array=grid_array)))

        for name, module in unet.named_modules():
            if type(module).__name__ == "CrossAttention" and "attn2" in name:
                module.mappings = mappings
            else:
                module.mappings = None
        return mappings

    return stablediffusion(use_grid_mappings_uncond, use_grid_mappings_cond,
                           unet=unet, vae=vae, clip=clip, device=device, clip_tokenizer=clip_tokenizer, **kwargs)


@torch.no_grad()
def stablediffusion(
    uncond_hook, cond_hook,
    *,
    unet,
    vae,
    clip,
    device,
    clip_tokenizer,
    guidance_scale=7.5,
    steps=50,
    seed=None,
    width=512,
    height=512,
    init_image=None,
    init_image_strength=0.5,
    batch_size=1,
) -> List[Image]:
    # Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64

    # If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None:
        seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)

    # Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )
    scheduler.set_timesteps(steps)

    # Preprocess image if it exists (img2img)
    if init_image is not None:
        # Resize and transpose for numpy b h w c -> torch b c h w
        init_image = init_image.resize(
            (width, height), resample=Image.Resampling.LANCZOS
        )
        init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
        init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

        # If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if init_image.shape[1] > 3:
            init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

        # Move image to GPU
        init_image = init_image.to(device)

        # Encode image
        with autocast(device):
            init_latent = (
                vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215
            )
        if batch_size != 1:
            # Just need to repeat the latent or something.
            raise NotImplementedError("Batch size > 1 is not supported yet for img2img.")

        t_start = steps - int(
            steps * init_image_strength
        )  # Nice, we start from image latent.
    else:
        init_latent = torch.zeros(
            (batch_size, unet.in_channels, height // 8, width // 8), device=device
        )
        t_start = 0

    # Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    # latent = noise * scheduler.init_noise_sigma
    latent = scheduler.add_noise(
        init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=device)
    ).to(device)

    # Process clip
    with autocast(device):
        dummy_emb = torch.zeros_like(_get_prompt_emb("", clip=clip, clip_tokenizer=clip_tokenizer, device=device)
            ).to(device)
        init_attention_func(
            unet=unet, clip=clip, vae=vae, device=device, clip_tokenizer=clip_tokenizer
        )
        timesteps = scheduler.timesteps[t_start:]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):
            t_index = t_start + i

            # sigma = scheduler.sigmas[t_index]
            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Predict the unconditional noise residual
            # use_unconditional_mappings()
            uncond_hook()
            noise_pred_uncond = unet(
                latent_model_input, t, encoder_hidden_states=dummy_emb,
            ).sample

            # Prepare the Cross-Attention layers
            # Predict the conditional noise residual and save the cross-attention layer activations
            # use_conditional_mappings()
            cond_hook()
            noise_pred_cond = unet(
                latent_model_input, t, encoder_hidden_states=dummy_emb,
            ).sample

            # Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )
            latent = scheduler.step(noise_pred, t_index, latent).prev_sample

        # scale and decode the image latents with vae
        latent = latent / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample

    assert batch_size >= 1
    # Same thing as before, but image is a batch of images
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    return [Image.fromarray(img) for img in image]


# Bounded to [0, 0.5]
def make_left_mask(W: int, H: int, percent=1.0) -> torch.Tensor:
    assert percent <= 1.0
    assert percent >= 0.0
    mask_left = torch.zeros(H, W, dtype=bool)
    mask_left[:, 0 : int(W // 2 * percent)] = 1
    return mask_left


# Bounded to [0, 0.5]
def make_right_mask(W: int, H: int, percent=1.0) -> torch.Tensor:
    assert percent <= 1.0
    assert percent >= 0.0
    mask_right = torch.zeros(H, W, dtype=bool)
    mask_right[:, int(W // 2 + (W // 2) * (1 - percent)) :] = 1
    return mask_right


# Bounded to [0, 0.5]
def make_top_mask(W: int, H: int, percent=1.0) -> torch.Tensor:
    assert percent <= 1.0
    assert percent >= 0.0
    mask_top = torch.zeros(H, W, dtype=bool)
    mask_top[0 : int(H // 2 * percent), :] = 1
    return mask_top


# Bounded to [0, 0.5]
def make_bottom_mask(W: int, H: int, percent=1.0) -> torch.Tensor:
    assert percent <= 1.0
    assert percent >= 0.0
    mask_bottom = torch.zeros(H, W, dtype=bool)
    mask_bottom[int(H // 2 + (H // 2) * (1 - percent)) :, :] = 1
    return mask_bottom


# Bounded to [0, 0.5]
def make_centre_horizontal_mask(W: int, H: int, percent=1.0) -> torch.Tensor:
    assert percent <= 1.0
    assert percent >= 0.0
    mask_centre_horizontal = torch.zeros(H, W, dtype=bool)
    mask_centre_horizontal[
        int(H / 2 - (H / 2) * (percent)) : int(H / 2 + (H / 2) * (percent)),
        :,
    ] = 1
    return mask_centre_horizontal


# Bounded to [0, 1]
def make_centre_vertical_mask(W: int, H: int, percent=1.0) -> torch.Tensor:
    assert percent <= 1.0
    assert percent >= 0.0
    mask_centre_vertical = torch.zeros(H, W, dtype=bool)
    mask_centre_vertical[
        :,
        int(W / 2 - (W / 2) * (percent)) : int(W / 2 + (W / 2) * (percent)),
    ] = 1
    return mask_centre_vertical


def make_true_mask(W: int, H: int) -> torch.Tensor:
    return torch.ones(H, W, dtype=bool)


def wrap_transpose(make_mask_fn) -> torch.Tensor:
    def inner(W: int, H: int):
        return torch.transpose(make_mask_fn(W, H), 0, 1)
    return inner


def make_grid_mask(W: int, H: int, grid_array: list, key: str, percent=0.0) -> torch.Tensor:
    """
    Given a grid_array of strings and None, like
    grid_array = [
        ["cat", None, "dog"],
        ["cat", None, "dog"],
        ["cat", None, "dog"],
    ],

    and a key "cat", we can break the image into different pieces.
    """
    mask = torch.zeros(H, W, dtype=bool)
    grid_h = len(grid_array)
    grid_w = len(grid_array[0])

    found = False

    for i, row in enumerate(grid_array):
        for j, token in enumerate(row):
            if token == key:
                found = True
                h_lo = int(i * H / grid_h)
                h_hi = int((i + 1) * H / grid_h)
                w_lo = int(j * W / grid_w)
                w_hi = int((j + 1) * W / grid_w)
                mask[h_lo:h_hi, w_lo:w_hi] = 1
    assert found, f"Key {key} not found in grid_array"
    return mask


def load_models():
    # Init CLIP tokenizer and model
    model_path_clip = "openai/clip-vit-large-patch14"
    clip_tokenizer = CLIPTokenizer.from_pretrained(model_path_clip)
    clip_model = CLIPModel.from_pretrained(model_path_clip, torch_dtype=torch.float16)
    clip = clip_model.text_model

    # Init diffusion model
    # Using Steven's huggingface auth token.
    auth_token = "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"  # Replace this with huggingface auth token as a string if model is not already downloaded
    model_path_diffusion = "CompVis/stable-diffusion-v1-4"
    unet = UNet2DConditionModel.from_pretrained(
        model_path_diffusion,
        subfolder="unet",
        use_auth_token=auth_token,
        revision="fp16",
        torch_dtype=torch.float16,
    )
    vae = AutoencoderKL.from_pretrained(
        model_path_diffusion,
        subfolder="vae",
        use_auth_token=auth_token,
        revision="fp16",
        torch_dtype=torch.float16,
    )

    # Move to GPU
    device = "cuda"
    unet.to(device)
    vae.to(device)
    clip.to(device)
    print("Loaded all models")

    return unet, vae, clip, clip_tokenizer, device


def main():
    unet, vae, clip, clip_tokenizer, device = load_models()
    # img = grid_exp(grid_array=[["cat", None, "dog"]],
    img = top_bottom_exp("cat", "dog",
        unet=unet,
        vae=vae,
        device=device,
        clip=clip,
        clip_tokenizer=clip_tokenizer,
        seed=248396402679,
        batch_size=2,
    )


if __name__ == "__main__":
    main()
