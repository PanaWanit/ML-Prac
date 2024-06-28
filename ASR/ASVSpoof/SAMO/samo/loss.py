import torch
from torch import nn
from torch.nn import functional as F

from typing import Tuple, Sequence, Optional, List


class SAMO(nn.Module):
    def __init__(
        self,
        m_real: float,
        m_fake: float,
        alpha: float,
    ) -> None:
        super().__init__()
        assert 0 <= m_real <= 1, "m_real must be in range [0, 1]"
        assert 0 <= m_fake <= 1, "m_fake must be in range [0, 1]"

        self.__m_real = m_real
        self.__m_fake = m_fake
        self.__alpha = alpha

    def forward(
        self, 
        x: torch.Tensor,
        labels: torch.Tensor,
        w_centers: torch.Tensor,
        w_spks: Optional[torch.Tensor] = None,
        get_score: bool = False
    ) -> torch.FloatTensor | Tuple[torch.FloatTensor, torch.Tensor]:
        """
        Args:
            x: (batch, feat_dim)
            labels(y): (batch)
            w_spks: a center of each speaker, for each embedding speaker corresponds to each utterance speaker (same shape as x);
        """
        x = F.normalize(x)
        w = F.normalize(w_centers).to(x.device)

        sim_scores = x @ w.transpose(0, 1)  # shape (batch, num_centers)
        max_scores, _ = sim_scores.max(dim=1)  # shape (batch)

        if w_spks is not None:  # attractor = True enrollment speakers)
            w_spks = F.normalize(w_spks).to(x.device)
            final_scores = torch.sum(w_spks * x, dim=1) # dot-product (row)vector-wise
        else:
            final_scores = max_scores

        # exponential part of eq(3).
        scores = torch.where(labels == 0, self.__m_real - final_scores, max_scores - self.__m_fake) 
        emb_loss = F.softplus(self.__alpha * scores).mean()
        if get_score:
            return emb_loss, final_scores
        return emb_loss
    
    # TODO: Inference function
    def inference(
        self, 
        x: torch.Tensor,
        labels: torch.Tensor,
        w_centers: torch.Tensor,
        w_spks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.Tensor]:
        raise NotImplementedError