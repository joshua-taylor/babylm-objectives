from .anyorder import AnyOrderLoss
from .latent import LatentLoss
from .samplers import build_sampler
from .token_losses import Loss, MTPLoss, NTPLoss, SelectiveLoss

LOSSES = {
    "ntp": NTPLoss,
    "mtp": MTPLoss,
    "selective": SelectiveLoss,
    "latent": LatentLoss,
    "anyorder": AnyOrderLoss,
}


def build_loss(name, model, corpus, args):
    if name not in LOSSES:
        raise ValueError(f"unknown loss {name!r}; choose from {list(LOSSES)}")
    return LOSSES[name](model, corpus, args)


__all__ = ["build_loss", "build_sampler", "LOSSES", "Loss"]
