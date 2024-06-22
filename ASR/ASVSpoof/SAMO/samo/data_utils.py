from typing import Any, List, Sequence, Dict, Tuple, TextIO
import numpy as np
from numpy.typing import ArrayLike
from aasist.data_utils import Dataset_ASVspoof2019_train

#################################################### genSpoof_list ####################################################
def _flatten2d(l2d:List[List[Any]]) -> List[Any]:
    return [x for l in l2d for x in l]

def _read_split(f: TextIO, enroll:bool) -> Sequence[np.ndarray]:
    lst = zip(*[line.strip().split(" ") for line in f.readlines()])
    if enroll: # len(lst) == 2 // (speakers, utt_string)
       spks, utt_list = np.array(lst[0]), [l.split(",") for l in lst[1]]
       num_utt_each_speaker = map(len, utt_list)

       spks = np.repeat(spks, num_utt_each_speaker)
       utt_list = _flatten2d(utt_list)
       return spks, utt_list

    return (np.array(l) for l in lst)
@np.vectorize
def _label_map(lb:str) -> int:
    return int(lb!="bonafide")


# extended from aasist.data_utils for samo dataset purpose

# TODO: test genSpoof_list on enrolled data
def genSpoof_list(
    dir_meta: str,
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
            mask = np.full_like(utt_list, True, dtype=np.bool_) if not target_only else np.isin(spks, enroll_spks)
            spks, utt_list, tag_list, labels = spks[mask], utt_list[mask], tag_list[mask], labels[mask]

    else: # for asv.{eval, dev}.trn protocols (enrolled data)
        with open(dir_meta, "r") as f:
            spks, utt_list = _read_split(f, enroll)
        tag_list = np.full_like(utt_list, "-")
        labels = np.zeros_like(utt_list) # no label for dev and eval set

    utt2spk = dict(zip(utt_list, spks))
    d_meta = dict(zip(utt_list, labels))
    return { ## key is mapped to keyword argument of ASVSpoof2019_speaker Dataset
        "base_dir": base_dir,
        "list_IDs": utt_list.tolist(), 
        "labels": d_meta, 
        "utt2spk": utt2spk, 
        "tag_list": tag_list.tolist(),
        "train": train
    }

########################################################################################################################

####################################################### Dataset ########################################################
class ASVspoof2019_speaker(Dataset_ASVspoof2019_train):
    def __init__(self, list_IDs, labels, utt2spk, base_dir, tag_list, train=True, cut=64600):
        super().__init__(list_IDs, labels, base_dir)
        self.utt2spk = utt2spk
        self.tag_list = tag_list
        self.cut = cut
        self.train = train
    def __getitem__(self, index):
        utt = self.list_IDs[index]
        tag = self.tag_list[index]
        x_inp, y = super().__getitem__(index)
        spk = self.utt2spk[utt]
        return x_inp, y, spk, utt, tag


if __name__ == '__main__':
    a,b,c,d = genSpoof_list('LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt', train=False, target_only=True)
    print(c, sep='\n')