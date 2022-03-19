import argparse
import os
import shutil
import time
import cv2
import numpy as np
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn

class logsigsum(torch.nn.Module):
    def __init__(self):
        super(logsigsum, self).__init__()
        self.fusion_func = torch.nn.LogSigmoid()

    def forward(self, activation_vision, activation_coord, activation_fusion):
        return self.fusion_func(activation_vision + activation_coord + activation_fusion)


class naivesum(torch.nn.Module):
    def __init__(self):
        super(naivesum, self).__init__()
        self.fusion_func = None

    def forward(self, activation_vision, activation_coord, activation_fusion):
        return activation_vision + activation_coord + activation_fusion

