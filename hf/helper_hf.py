import argparse

from huggingface_hub import PyTorchModelHubMixin, HfApi
import torch

from mirage_hf import MIRAGEWrapper


class MIRAGEhf(MIRAGEWrapper, PyTorchModelHubMixin):
    def __init__(
        self,
        input_size=512,
        patch_size=32,
        modalities='bscan-slo',
        size='base',
    ):
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            modalities=modalities,
            size=size,
        )

api = HfApi()


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--task', type=str, choices=['push', 'load'])
parser.add_argument('-s', '--size', type=str, choices=['base', 'large'])
args = parser.parse_args()


if args.task == 'push':
    config={
        "input_size": 512,
        "patch_size": 32,
        "modalities": "bscan-slo",
    }
    if args.size == 'large':
        config["size"] = "large"
        weights = "../__weights/MIRAGE-Large.pth"
        name = "MIRAGE-Large"
    else:
        config["size"] = "base"
        weights = "../__weights/MIRAGE-Base.pth"
        name = "MIRAGE-Base"
    state_dict = torch.load(weights, weights_only=False)['model']
    model = MIRAGEhf(**config)
    msg = model.load_state_dict(state_dict, strict=False)
    print('  # Missing keys:', len(msg.missing_keys))
    print('  # Unexpected keys:', len(msg.unexpected_keys))
    assert len(msg.missing_keys) == 0
    # save locally
    model.save_pretrained(name, config=config)
    # push to the hub
    model.push_to_hub(name, config=config)
    # Also push the README file
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=f"j-morano/{name}",
        repo_type="model",
    )

elif args.task == 'load':
    if args.size == 'large':
        name = "MIRAGE-Large"
    else:
        name = "MIRAGE-Base"
    model = MIRAGEhf.from_pretrained(f"j-morano/{name}")

    img = torch.randn(1, 1, 512, 512)

    model.eval()
    with torch.no_grad():
        out = model({'bscan': img, 'slo': img})
        print(out.shape)

