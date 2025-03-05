import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from imports.ParametersManager import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from Ablationnobyerandtran import ManTraNet
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
def pixel_level_auc(out, label,threshold=0.5):
    out=out.view(-1).cpu().detach().numpy()
    label=label.view(-1).cpu().detach().numpy()
    re=roc_auc_score(label, out)
    return re
def pixel_level_f1(out, label,threshold=0.5):
    out=out.view(-1).cpu().detach().numpy()
    label=label.view(-1).cpu().detach().numpy()
    re1 = f1_score(label,
                   out,
                   labels=None,
                   pos_label=1,

                   sample_weight=None)
    return re1
