import os
import sys

import torch
import torchvision.models as tv_models


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from misc.utils import load_model as load_builtin_expert  # noqa: E402


def _build_torchvision_model(model_name, num_classes):
    try:
        return tv_models.get_model(model_name, weights=None, num_classes=num_classes)
    except Exception:
        if not hasattr(tv_models, model_name):
            raise ValueError(f"Unknown torchvision expert model: {model_name}")
        model_fn = getattr(tv_models, model_name)
        try:
            return model_fn(weights=None, num_classes=num_classes)
        except TypeError:
            return model_fn(pretrained=False, num_classes=num_classes)


def _state_dict_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Expert checkpoint does not contain a valid state_dict.")

    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }


def load_expert_model(model_name, dataset, num_classes, ckpt_path=None, pretrained=True):
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Expert checkpoint not found: {ckpt_path}")
        model = _build_torchvision_model(model_name, num_classes)
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(_state_dict_from_checkpoint(checkpoint), strict=False)
        if missing or unexpected:
            print(
                f"[expert_loader] Loaded {ckpt_path} with "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys."
            )
        return model

    return load_builtin_expert(
        model_name=model_name,
        dataset=dataset,
        pretrained=pretrained,
        classes=range(num_classes),
    )
