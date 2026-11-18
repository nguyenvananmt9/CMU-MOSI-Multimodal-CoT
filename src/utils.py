import os
import re
import gc
import copy
import time
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import BertModel, AutoModelForCausalLM, AutoTokenizer


DATA_PATH = "/content/drive/MyDrive/CMU-MOSI/Processed/Processed/unaligned_50.pkl"
SAVE_DIR = "./results/predictions"
os.makedirs(SAVE_DIR, exist_ok=True)

with open(DATA_PATH, "rb") as f:
    data = pickle.load(f, encoding="latin1")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | PyTorch: {torch.__version__}")
print(f"Splits: Train={len(data['train']['regression_labels'])}, Valid={len(data['valid']['regression_labels'])}, Test={len(data['test']['regression_labels'])}")

def set_seed(seed=42, deterministic=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)

def get_feature_loaders(data_dict, batch_size=128, seed=1115, use_generator=True):
    def _make_dataset(split):
        t = torch.tensor(np.nan_to_num(np.asarray(data_dict[split]["text"], dtype=np.float32)))
        a = torch.tensor(np.nan_to_num(np.asarray(data_dict[split]["audio"], dtype=np.float32)))
        v = torch.tensor(np.nan_to_num(np.asarray(data_dict[split]["vision"], dtype=np.float32)))
        y = torch.tensor(np.nan_to_num(np.asarray(data_dict[split]["regression_labels"], dtype=np.float32))).reshape(-1, 1)
        return TensorDataset(t, a, v, y)

    gen = torch.Generator().manual_seed(seed) if use_generator else None
    train_loader = DataLoader(_make_dataset("train"), batch_size=batch_size, shuffle=True, generator=gen)
    valid_loader = DataLoader(_make_dataset("valid"), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(_make_dataset("test"),  batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader

class BertMOSIDataset(Dataset):
    def __init__(self, split_data):
        self.text_bert = torch.as_tensor(np.asarray(split_data["text_bert"]), dtype=torch.long)
        self.audio = torch.as_tensor(np.asarray(split_data["audio"]), dtype=torch.float32)
        self.vision = torch.as_tensor(np.asarray(split_data["vision"]), dtype=torch.float32)
        self.labels = torch.as_tensor(np.asarray(split_data["regression_labels"]), dtype=torch.float32).reshape(-1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.text_bert[idx], self.audio[idx], self.vision[idx], self.labels[idx]

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    true_has0 = (y_true >= 0).astype(int)
    pred_has0 = (y_pred >= 0).astype(int)
    has0_acc = accuracy_score(true_has0, pred_has0) * 100
    has0_f1  = f1_score(true_has0, pred_has0, average="binary", zero_division=0) * 100

    mask = y_true != 0
    true_non0 = (y_true[mask] > 0).astype(int)
    pred_non0 = (y_pred[mask] > 0).astype(int)
    non0_acc = accuracy_score(true_non0, pred_non0) * 100
    non0_f1  = f1_score(true_non0, pred_non0, average="binary", zero_division=0) * 100

    mae  = mean_absolute_error(y_true, y_pred)
    corr = pearsonr(y_true, y_pred)[0] if (np.std(y_true) > 0 and np.std(y_pred) > 0) else np.nan

    return {
        "Has0_Acc2": has0_acc,
        "Has0_F1": has0_f1,
        "Non0_Acc2": non0_acc,
        "Non0_F1": non0_f1,
        "MAE": mae,
        "Corr": corr
    }