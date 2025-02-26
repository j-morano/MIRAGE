import re

import torch



def interpolate_pos_embed(model, checkpoint_model):
    pattern = r"input_adapters\.(.*)\.pos_emb"
    matched_keys = [k for k in checkpoint_model if bool(re.match(pattern, k))]

    for key in matched_keys:
        # group(0) is entire matched regex
        domain = re.match(pattern, key).group(1)  # type: ignore
        if getattr(model.input_adapters, domain, None) is not None:
            pos_embed_checkpoint = checkpoint_model[key]
            _, _, orig_H, orig_W = pos_embed_checkpoint.shape
            _, _, new_H, new_W = getattr(model.input_adapters, domain).pos_emb.shape
            if (orig_H != new_H) or (orig_W != new_W):
                print(f"Key {key}: Position interpolate from {orig_H}x{orig_W} to {new_H}x{new_W}")
                pos_embed_checkpoint = torch.nn.functional.interpolate(
                    pos_embed_checkpoint, size=(new_H, new_W), mode='bicubic', align_corners=False)
                checkpoint_model[key] = pos_embed_checkpoint
