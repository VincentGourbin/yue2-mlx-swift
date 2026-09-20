"""Regenerates every committed parity/ fixture from the upstream yue2 classes (fiche
T-1.3). Run from the repo root: `.venv-ref/bin/python Scripts/reference/tiny_fixtures.py`.
"""
import torch

import tiny_lm
import tiny_protocol
import tiny_sampling
import tiny_vae

if __name__ == "__main__":
    torch.set_num_threads(1)
    tiny_vae.save()
    tiny_lm.save()
    tiny_sampling.save()
    tiny_protocol.save()
    print("tiny fixtures written to parity/")
