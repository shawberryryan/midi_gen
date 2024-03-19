"""
Utility functions adapted from EECS 445 proj 2
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import re
import torch


def load_latest_model(pattern):
    # List all files matching the pattern
    files = glob.glob(pattern)

    # Extract numbers from filenames and find the highest one
    max_num = -1
    latest_file = None
    for file in files:
        # Extract number using regular expression
        match = re.search(r"vae_saved_(\d+)\.pt", file)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                latest_file = file

    # Load and return the latest model
    if latest_file:
        print(f"Loading model from {latest_file}")
        model = torch.load(latest_file)
        return model, latest_file
    else:
        print("No model files found.")
        return None, None
