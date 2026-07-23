import fvdb
import fvdb.nn as fvnn
import torch


def grid_to_VDB(grid: fvdb.GridBatch, torch_func=torch.zeros, additional_feat=[], dtype=torch.float32):
    size = list(grid.jidx.shape)
    size += additional_feat
    tensor_feature = torch_func(size, dtype=dtype, device=grid.device)
    return fvnn.VDBTensor(grid, grid.jagged_like(tensor_feature))
