import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple, Sequence, Optional, List


class SAMO(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        m_real: float,
        m_fake: float,
        alpha: float,
        num_centers: Sequence[int],
        initialize_centers: str,
    ) -> None:
        super().__init__()
        assert 0 <= m_real <= 1, "m_real must be in range [0, 1]"
        assert 0 <= m_fake <= 1, "m_fake must be in range [0, 1]"

        self.__feat_dim = feat_dim
        self.__num_centers = num_centers
        self.__m_real = m_real
        self.__m_fake = m_fake
        self.__alpha = alpha

        if initialize_centers == "one_hot":
            self.__w_centers = torch.eye(feat_dim)[:num_centers]
        elif initialize_centers == "evenly":  # uniform_hypersphere
            raise NotImplementedError()
        else:
            raise RuntimeError("There is no {initialize_centers} method.")

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor, w_spks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.FloatTensor, torch.Tensor]:
        """
        Args:
            x: (batch, feat_dim)
            labels(y): (batch)
            w_spks: a center of each speaker, for each embedding speaker corresponds to each utterance speaker (same shape as x);
        """
        x = F.normalize(x)
        w = F.normalize(self.w_centers).to(x.device)

        sim_scores = x @ w.transpose(0, 1)  # shape (batch, num_centers)
        max_scores, _ = sim_scores.max(dim=1)  # shape (batch)

        if w_spks is not None:  # attractor = True enrollment speakers)
            w_spks = F.normalize(w_spks).to(x.device)
            final_scores = torch.sum(w_spks * x, dim=1) # dot-product (row)vector-wise
        else:
            final_scores = max_scores

        # exponential part of eq(3).
        scores = torch.where(labels == 0, self.m_real - final_scores, max_scores - self.m_fake) 
        emb_loss = F.softplus(self.alpha * scores).mean()

        return emb_loss, final_scores
    
    # TODO: Inference function
    def inference(self, x, labels, w_spks):
        raise NotImplementedError
    
    # Getters and Setters
    @property
    def m_real(self): return self.__m_real
    @property
    def m_fake(self): return self.__m_fake
    @property
    def alpha(self): return self.__alpha
    @property
    def num_centers(self): return self.__num_centers
    @property
    def feat_dim(self): return self.__feat_dim

    @property
    def w_centers(self): return self.__w_centers
    @w_centers.setter
    def set_centers(self, l: torch.Tensor | List[torch.Tensor]):
        if isinstance(l, list):
            l = torch.stack(l)
        self.__w_centers = l
