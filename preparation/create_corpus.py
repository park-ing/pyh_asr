import tarfile

# PyTorch
import torch
import torch.nn as nn
import torchvision
import torchaudio
from torchvision.datasets.utils import extract_archive

# Other
import os
import glob
from tqdm import tqdm
import sentencepiece as spm
import numpy as np
import requests
import pickle
import gdown


def main():

    root = '/home/nas4/DB/AIHUB_track2_2/data/Validation/medv'
    version = 'doc'
    mode = "validation"

    corpus_path = os.path.join("/home/nas4/user/yh/ai_hub_korean/spm", "corpus_val_{}.txt".format(version))

    print("Create Corpus File:", mode)
    corpus_file = open(corpus_path, "w")


    for file_path in tqdm(glob.glob(os.path.join(root, version, "*", "*.txt"))):
        with open(file_path, "r") as f:
            #line = f.readline()[7:].replace("{NS}", "").replace("{LG}", "").lower()
            line = f.readline().rstrip()
            line += "\n"
            corpus_file.write(line)

if __name__ =='__main__':
    main()