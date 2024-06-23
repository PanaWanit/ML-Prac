from typing import Any, List, Sequence, Dict, Tuple, TextIO
import os
from io import StringIO
import soundfile as sf

import numpy as np
from numpy.typing import ArrayLike

from aasist.data_utils import pad, pad_random

from torch import Tensor
from torch.utils.data import Dataset, Subset

#################################################### genSpoof_list ####################################################
def _flatten2d(l2d:List[List[Any]]) -> np.ndarray[Any]:
    return np.array([x for l in l2d for x in l])

def _read_split(f: TextIO, enroll:bool) -> Sequence[np.ndarray]:
    lst = list(zip(*[line.strip().split(" ") for line in f.readlines()]))
    if enroll:
        assert len(lst) == 2
        spks, utt_list = np.array(lst[0]), [l.split(",") for l in lst[1]]
        num_utt_each_speaker = list(map(len, utt_list))

        spks = np.repeat(spks, num_utt_each_speaker)
        utt_list = _flatten2d(utt_list)
        return spks, utt_list

    return (np.array(l) for l in lst)

@np.vectorize
def _label_map(lb:str) -> int:
    return int(lb!="bonafide")

def _concat_text_files(dir_meta: List[str]):
    txt = ""
    for dir in dir_meta:
        with open(dir, "r") as f:
            txt += f.read()
    return StringIO(txt)

# read from asv_protocols/LA.asv.{task}.{male, female}.trn
def get_enroll_speaker(dir_meta: List[str] | str): 
    spks, _ = _read_split(_concat_text_files(dir_meta), enroll=False)
    return list(set(spks)) # cast to list to make it compatible with np.isin with assume_unique=True
    


# extended from aasist.data_utils for samo dataset purposes and make it compatible with the dataset.
def genSpoof_list(
    dir_meta: str | List[str],
    base_dir: str,
    enroll: bool = False,
    train: bool = True,
    target_only: bool = False,
    enroll_spks: None | Sequence[str] = None,
) -> Dict[Any, Any]:
    if not enroll:
        with open(dir_meta, "r") as f:
            spks, utt_list, _, tag_list, labels = _read_split(f, enroll)
            labels = _label_map(labels)

        if not train:  # dev, eval
            mask = np.full_like(utt_list, True, dtype=np.bool_) if not target_only else np.isin(spks, enroll_spks, assume_unique=True)
            spks, utt_list, tag_list, labels = spks[mask], utt_list[mask], tag_list[mask], labels[mask]

    else: # for asv.{eval, dev}.trn protocols (enrolled data)
        f = _concat_text_files(dir_meta) # enrolled data contain 2 utterance annotated files for each task, (i.e. female, male)
        spks, utt_list = _read_split(f, enroll)
        tag_list = np.full_like(utt_list, "-")
        labels = np.zeros_like(utt_list) # no label for dev and eval set

    utt2spk = dict(zip(utt_list, spks))
    d_meta = dict(zip(utt_list, labels))
    return { # key is mapped to the keyword argument of the ASVSpoof2019_speaker dataset.
        "base_dir": base_dir,
        "list_IDs": utt_list.tolist(), 
        "labels": d_meta, 
        "utt2spk": utt2spk, 
        "tag_list": tag_list.tolist(),
        "train": train
    }

########################################################################################################################

####################################################### Dataset ########################################################
# combined Dataset_ASVspoof2019_train and Dataset_ASVspoof2019_devNeval into one Dataset + SAMO's purpose
class ASVspoof2019_speaker(Dataset):
    def __init__(self, list_IDs, labels, utt2spk, base_dir, tag_list, train=True, cut=64600):
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.utt2spk = utt2spk
        self.tag_list = tag_list
        self.cut = cut
        self.pad = pad_random if train else pad
    
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        utt = self.list_IDs[index]
        tag = self.tag_list[index]
        spk = self.utt2spk[utt]

        X, _ = sf.read(os.path.join(self.base_dir, f"flac/{utt}.flac"))
        X_pad = self.pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[utt]

        return x_inp, y, spk, utt, tag

def subset_bonafide(dataset: ASVspoof2019_speaker) -> ASVspoof2019_speaker:
    bonafide_index = [i for i in range(len(dataset)) if dataset.labels[dataset.list_IDs[i]]==0]
    return Subset(dataset, bonafide_index) # bonafide = 0, spoof = 1