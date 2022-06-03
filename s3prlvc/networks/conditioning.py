"""Network modules of Conditioning"""


from dataclasses import dataclass

from torch import nn, Tensor, cat # pylint: disable=no-name-in-module
import torch.nn.functional as F
from omegaconf import MISSING


@dataclass
class ConfGlobalCondNet:
    """
    Args:
        integration_type - Type of input-conditioning integration ("add" | "concat")
        dim_io - Dimension size of input/output
        dim_global_cond - Dimension size of global conditioning vector
    """
    integration_type: str = MISSING
    dim_io: int = MISSING
    dim_global_cond: int = MISSING

class GlobalCondNet(nn.Module):
    """Global conditioning module of Taco2AR.

    Add    mode: o_series = i_series + project(expand(cond_vec))
    Concat mode: o_series = project(cat(i_series, expand(cond_vec)))
    """

    def __init__(self, conf: ConfGlobalCondNet):
        super().__init__()
        self._integration_type = conf.integration_type

        # Determine dimension size of integration product
        if conf.integration_type == "add":
            assert conf.dim_global_cond == conf.dim_io
            # [Batch, T_max, hidden] => [Batch, T_max, hidden==emb] => [Batch, T_max, hidden]
            dim_integrated = conf.dim_io
        elif conf.integration_type == "concat":
            # [Batch, T_max, hidden] => [Batch, T_max, hidden+emb] => [Batch, T_max, hidden]
            dim_integrated = conf.dim_io + conf.dim_global_cond
        else:
            raise ValueError(f"Integration type '{conf.integration_type}' is not supported.")

        self.projection = nn.Linear(dim_integrated, conf.dim_io)

    # Typing of PyTorch forward API is poor.
    def forward(self, i_series: Tensor, global_cond_vec: Tensor) -> Tensor: # pyright: reportIncompatibleMethodOverride=false
        """Integrate a global conditioning vector with an input series.

        Args:
            i_series (Batch, T, dim_i) - input series
            global_cond_vec (Batch, dim_global_cond) - global conditioning vector
        Returns:
            Integrated series
        """

        global_cond_normed = F.normalize(global_cond_vec)

        if self._integration_type == "add":
            return i_series + self.projection(global_cond_normed).unsqueeze(1)
        elif self._integration_type == "concat":
            cond_series = global_cond_normed.unsqueeze(1).expand(-1, i_series.size(1), -1)
            return self.projection(cat([i_series, cond_series], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")
