"""Run the two-tower model in a child process and get its vectors back as numpy arrays.

On macOS, torch and LightGBM each load their own OpenMP runtime (torch bundles libomp, LightGBM links
Homebrew's). Two runtimes in one process segfault or deadlock, whichever is imported first. So main.py
never imports torch: training happens in a fresh interpreter running two_tower.py, and only arrays come back.
This module must not import torch.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np


def vectors_in_subprocess(train, n_users, n_items, feats):
    """Returns (user_vecs [n_users, d], item_vecs [n_items, d]) from a two-tower trained in a child process."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "two_tower.py")
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npz")
        np.savez(inp, u=train["u"].values, i=train["i"].values, n_users=n_users, n_items=n_items,
                 mh=feats["mh"], price=feats["price"], year=feats["year"])
        subprocess.run([sys.executable, script, inp, out], check=True)
        with np.load(out) as r:
            return r["user_vecs"], r["item_vecs"]
