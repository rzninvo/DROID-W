from typing import Dict, List, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from src.utils.sys_timer import timer

"""
From FiT3D, here we subclass the model instead of overriding the "get_intermediate_layers" method
as it will cause errors in multipprocessing setup of the SLAM system
"""


class Fit3DModels(torch.nn.Module):
    def __init__(self, extractor_model, device):
        super().__init__()
        self.model = torch.hub.load("ywyue/FiT3D", extractor_model).to(device).eval()

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n=1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        # Two paths: legacy timm exposed `_intermediate_layers` returning
        # full token sequences (prefix + patch) which we then split. timm
        # 1.0+ removed that; the public `get_intermediate_layers` accepts
        # `return_prefix_tokens=True` and returns tuples of (patch, prefix).
        if hasattr(self.model, "_intermediate_layers"):
            full = self.model._intermediate_layers(x, n)
            if norm:
                full = [self.model.norm(out) for out in full]
            if return_class_token:
                prefix_tokens = [out[:, 0] for out in full]
            else:
                prefix_tokens = [
                    out[:, 0 : self.model.num_prefix_tokens] for out in full
                ]
            outputs = [out[:, self.model.num_prefix_tokens :] for out in full]
        else:
            tuples = self.model.get_intermediate_layers(
                x, n=n, reshape=False, return_prefix_tokens=True, norm=norm,
            )
            outputs = [pair[0] for pair in tuples]   # patch tokens
            full_prefix = [pair[1] for pair in tuples]  # (B, num_prefix_tokens, C)
            if return_class_token:
                prefix_tokens = [pt[:, 0] for pt in full_prefix]
            else:
                prefix_tokens = full_prefix

        if reshape:
            B, C, H, W = x.shape
            grid_size = (
                (H - self.model.patch_embed.patch_size[0])
                // self.model.patch_embed.proj.stride[0]
                + 1,
                (W - self.model.patch_embed.patch_size[1])
                // self.model.patch_embed.proj.stride[1]
                + 1,
            )
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]

        if return_prefix_tokens or return_class_token:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)


"""
Done with overwriting get_intermediate_layers of FiT3D model
"""


def _create_dinov3_model(extractor_model: str, device: str) -> nn.Module:
    """Load DINOv3 ViT-S/16 from torch hub with HuggingFace weights."""
    import os
    hub_dir = os.path.expanduser("~/.cache/torch/hub/facebookresearch_dinov3_main")

    # Download hub repo if not cached
    if not os.path.isdir(hub_dir):
        torch.hub.load("facebookresearch/dinov3", extractor_model, pretrained=False)

    # Build model without pretrained weights
    model = torch.hub.load(hub_dir, extractor_model, source="local", pretrained=False)

    # Load converted checkpoint (HF safetensors -> torch hub format)
    ckpt_path = os.path.expanduser(
        "~/.cache/torch/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )
    if not os.path.isfile(ckpt_path):
        # Convert HuggingFace safetensors to torch hub format on first run
        _convert_dinov3_hf_to_hub(model, ckpt_path)
    else:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)

    return model.to(device).eval()


def _convert_dinov3_hf_to_hub(model: nn.Module, save_path: str) -> None:
    """Convert DINOv3 HuggingFace safetensors to torch hub state dict format."""
    import os
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download

    hf_path = hf_hub_download(
        "facebook/dinov3-vits16-pretrain-lvd1689m", "model.safetensors"
    )
    hf_state = load_file(hf_path)
    hub_state = model.state_dict()
    new_state = {}

    # Embeddings
    new_state["cls_token"] = hf_state["embeddings.cls_token"]
    new_state["patch_embed.proj.weight"] = hf_state["embeddings.patch_embeddings.weight"]
    new_state["patch_embed.proj.bias"] = hf_state["embeddings.patch_embeddings.bias"]
    new_state["storage_tokens"] = hf_state["embeddings.register_tokens"]
    new_state["norm.weight"] = hf_state["norm.weight"]
    new_state["norm.bias"] = hf_state["norm.bias"]
    if "mask_token" in hub_state:
        mt = hf_state["embeddings.mask_token"]
        if hub_state["mask_token"].shape != mt.shape:
            mt = mt.squeeze(0)
        new_state["mask_token"] = mt

    # Transformer blocks
    num_layers = sum(1 for k in hf_state if k.startswith("layer.") and k.endswith(".norm1.weight"))
    for i in range(num_layers):
        q_w = hf_state[f"layer.{i}.attention.q_proj.weight"]
        k_w = hf_state[f"layer.{i}.attention.k_proj.weight"]
        v_w = hf_state[f"layer.{i}.attention.v_proj.weight"]
        new_state[f"blocks.{i}.attn.qkv.weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        q_b = hf_state[f"layer.{i}.attention.q_proj.bias"]
        k_b = hf_state.get(f"layer.{i}.attention.k_proj.bias", torch.zeros_like(q_b))
        v_b = hf_state[f"layer.{i}.attention.v_proj.bias"]
        new_state[f"blocks.{i}.attn.qkv.bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        new_state[f"blocks.{i}.attn.proj.weight"] = hf_state[f"layer.{i}.attention.o_proj.weight"]
        new_state[f"blocks.{i}.attn.proj.bias"] = hf_state[f"layer.{i}.attention.o_proj.bias"]
        new_state[f"blocks.{i}.ls1.gamma"] = hf_state[f"layer.{i}.layer_scale1.lambda1"]
        new_state[f"blocks.{i}.ls2.gamma"] = hf_state[f"layer.{i}.layer_scale2.lambda1"]
        new_state[f"blocks.{i}.mlp.fc1.weight"] = hf_state[f"layer.{i}.mlp.up_proj.weight"]
        new_state[f"blocks.{i}.mlp.fc1.bias"] = hf_state[f"layer.{i}.mlp.up_proj.bias"]
        new_state[f"blocks.{i}.mlp.fc2.weight"] = hf_state[f"layer.{i}.mlp.down_proj.weight"]
        new_state[f"blocks.{i}.mlp.fc2.bias"] = hf_state[f"layer.{i}.mlp.down_proj.bias"]
        new_state[f"blocks.{i}.norm1.weight"] = hf_state[f"layer.{i}.norm1.weight"]
        new_state[f"blocks.{i}.norm1.bias"] = hf_state[f"layer.{i}.norm1.bias"]
        new_state[f"blocks.{i}.norm2.weight"] = hf_state[f"layer.{i}.norm2.weight"]
        new_state[f"blocks.{i}.norm2.bias"] = hf_state[f"layer.{i}.norm2.bias"]

    # Copy non-learned buffers (bias_mask, rope_embed)
    for k in set(hub_state.keys()) - set(new_state.keys()):
        new_state[k] = hub_state[k]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(new_state, save_path)
    model.load_state_dict(new_state, strict=False)


def get_feature_extractor(cfg: Dict) -> nn.Module:
    """
    Get the feature extractor model based on the configuration.
    """
    device = cfg["device"]
    extractor_model = cfg["mono_prior"]["feature_extractor"]

    if extractor_model in ["dinov2_reg_small_fine", "dinov2_small_fine"]:
        return Fit3DModels(extractor_model, device)
    elif extractor_model in ["dinov2_vits14", "dinov2_vits14_reg"]:
        return (
            torch.hub.load("facebookresearch/dinov2", extractor_model).to(device).eval()
        )
    elif extractor_model in ["dinov3_vits16", "dinov3_vits16plus"]:
        return _create_dinov3_model(extractor_model, device)
    else:
        # If use other feature extractor as prior, add code here
        raise NotImplementedError("Unsupported feature extractor")


@torch.no_grad()
@timer.section("DINO Feature Extraction")
def predict_img_features(
    model: nn.Module,
    idx: int,
    input_tensor: torch.Tensor,
    cfg: Dict,
    device: str,
    save_feat: bool = True,
    suffix: str = "",
) -> torch.Tensor:
    """
    Predict image features using the given model.

    Args:
        model (nn.Module): The feature extractor model.
        idx (int): Image index.
        input_tensor (torch.Tensor): Input image tensor of shape (1, 3, H, W).
        cfg (Dict): Configuration dictionary.
        device (str): Device to run the model on.
        save_feat (bool): Whether to save the features.
        suffix (str): Suffix for the output file name.

    Returns:
        torch.Tensor: Extracted features.
    """
    extractor_model = cfg["mono_prior"]["feature_extractor"]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    normalize = transforms.Normalize(mean=mean, std=std)

    if extractor_model in ["dinov3_vits16", "dinov3_vits16plus"]:
        stride = 16
        image_resized = process_image(input_tensor, stride, normalize, device)
    else:
        stride = 14
        image_resized = process_image(input_tensor, stride, normalize, device)

    if extractor_model in ["dinov2_reg_small_fine", "dinov2_small_fine"]:
        features = model.get_intermediate_layers(
            image_resized,
            n=[8, 9, 10, 11],
            reshape=True,
            return_prefix_tokens=False,
            return_class_token=False,
            norm=True,
        )
        features = features[-1].squeeze().permute((1, 2, 0))
    elif extractor_model in ["dinov2_vits14", "dinov2_vits14_reg"]:
        features_dict = model.forward_features(image_resized)
        features = features_dict["x_norm_patchtokens"].view(
            image_resized.shape[2] // 14, image_resized.shape[3] // 14, -1
        )
    elif extractor_model in ["dinov3_vits16", "dinov3_vits16plus"]:
        features_dict = model.forward_features(image_resized)
        features = features_dict["x_norm_patchtokens"].view(
            image_resized.shape[2] // 16, image_resized.shape[3] // 16, -1
        )
    else:
        # If use other feature extractor as prior, add code here
        raise NotImplementedError("Unsupported feature extractor")

    if save_feat:
        _save_features(features, cfg, idx, suffix)

    return features


def process_image(
    image: torch.Tensor, stride: int, transforms: nn.Module, device: str = "cuda"
) -> torch.Tensor:
    """
    Process the input image for feature extraction.

    Args:
        image (torch.Tensor): Input image tensor.
        stride (int): Stride for resizing.
        transforms (nn.Module): Normalization transforms.
        device (str): Device to run the processing on.

    Returns:
        torch.Tensor: Processed image tensor.
    """
    image_tensor = transforms(image).float().to(device)
    h, w = image_tensor.shape[2:]
    height_int = (h // stride) * stride
    width_int = (w // stride) * stride
    return F.interpolate(image_tensor, size=(height_int, width_int), mode="bilinear")


def _save_features(features: torch.Tensor, cfg: Dict, idx: int, suffix: str) -> None:
    """
    Save the extracted features to a file.

    Args:
        features (torch.Tensor): Extracted features.
        cfg (Dict): Configuration dictionary.
        idx (int): Image index.
        suffix (str): Suffix for the output file name.
    """
    output_dir = f"{cfg['data']['output']}/{cfg['scene']}"
    output_path = f"{output_dir}/mono_priors/features/{idx:05d}{suffix}.npy"
    final_feat = features.detach().cpu().float().numpy()
    np.save(output_path, final_feat)
