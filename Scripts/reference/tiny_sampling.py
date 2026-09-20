"""Sampling fixture (plan/08-tests.md): shared random logits [1, 184704] (seed 7) run
through the upstream yue2.sampling.distribution() under 6 configs. No model weights."""
import json

import torch
from yue2.protocol import EOD, CODEC_OFFSET, CODEC_SIZE, LATENT_START, Sampling
from yue2.sampling import distribution

from _common import META, save_file_deterministic

ABC_DEFAULT = Sampling(0.7, 0.9, 30, 1.005, 100, 32, 4096)
SEMANTIC_DEFAULT = Sampling()


def _history(generator, low, high, count):
    return torch.randint(low, high, (count,), generator=generator).tolist()


def build():
    generator = torch.Generator().manual_seed(7)
    logits = torch.randn(1, 184704, generator=generator)
    out = {"logits": logits}
    configs = [
        dict(name="cfg1", sampling=ABC_DEFAULT, phase="abc", step=40,
             history=_history(generator, 0, EOD, 60), legacy_off=False, dtype=torch.float32),
        dict(name="cfg2", sampling=SEMANTIC_DEFAULT, phase="semantic", step=300,
             history=_history(generator, CODEC_OFFSET, LATENT_START, 70), legacy_off=False, dtype=torch.float32),
        dict(name="cfg3", sampling=Sampling(temperature=0, top_p=.95, top_k=100, repetition_penalty=1.2,
                                            penalty_window=50, min_tokens=200, max_tokens=9000),
             phase="semantic", step=300,
             history=_history(generator, CODEC_OFFSET, LATENT_START, 70), legacy_off=False, dtype=torch.float32),
        dict(name="cfg4", sampling=Sampling(temperature=1., top_p=1., top_k=100, repetition_penalty=1.2,
                                            penalty_window=50, min_tokens=200, max_tokens=9000),
             phase="semantic", step=300,
             history=_history(generator, CODEC_OFFSET, LATENT_START, 70), legacy_off=False, dtype=torch.float32),
        dict(name="cfg5", sampling=SEMANTIC_DEFAULT, phase="semantic", step=300,
             history=_history(generator, CODEC_OFFSET, LATENT_START, 70), legacy_off=True, dtype=torch.bfloat16),
        dict(name="cfg6", sampling=SEMANTIC_DEFAULT, phase="semantic", step=3,
             history=_history(generator, CODEC_OFFSET, LATENT_START, 10), legacy_off=False, dtype=torch.float32),
    ]
    metadata = {**META, "kind": "sampling", "seed": "7"}
    for config in configs:
        name = config["name"]
        source = logits.to(config["dtype"])
        history = config["history"]
        scores = distribution(source, config["sampling"], history, config["step"], config["phase"], config["legacy_off"])
        out[f"{name}_history"] = torch.tensor(history, dtype=torch.int32)
        out[f"{name}_step"] = torch.tensor([config["step"]], dtype=torch.int32)
        # `scores` is [1, 184704] and mostly -inf after phase/top-k masking; store it as
        # a lossless sparse (index, value) pair over the finite entries only (VOCAB_SIZE
        # implicit, all other positions are exactly -inf) to keep parity/ under budget.
        finite = torch.isfinite(scores[0])
        out[f"{name}_finite_indices"] = finite.nonzero(as_tuple=True)[0].to(torch.int32)
        out[f"{name}_finite_values"] = scores[0][finite]
        out[f"{name}_argmax"] = scores.argmax(-1).to(torch.int32)
        metadata[name] = json.dumps({
            "phase": config["phase"], "legacy_off": config["legacy_off"], "dtype": str(config["dtype"]),
            "temperature": config["sampling"].temperature, "top_p": config["sampling"].top_p,
            "top_k": config["sampling"].top_k, "repetition_penalty": config["sampling"].repetition_penalty,
            "penalty_window": config["sampling"].penalty_window, "min_tokens": config["sampling"].min_tokens,
            "max_tokens": config["sampling"].max_tokens,
        })
    return out, metadata


def save(path="parity/sampling.safetensors"):
    tensors, metadata = build()
    save_file_deterministic(tensors, path, metadata=metadata)


if __name__ == "__main__":
    save()
