from .var_man import BaseVarMan
import torch
import copy


class TorchVariableManage(BaseVarMan):
    def __init__(self, model):
        super().__init__()
        self.var_list = [p for p in model.parameters()]

    def pull_val(self, val_idx: list):
        for i in val_idx:
            self.var_list[i].detach().numpy()[:] = self.val_list[i]

    def var_tolist(self, vars):
        # Note : need deep copy , nor the parameters will be modified before pull_val
        return [copy.deepcopy(v.detach().numpy()) for v in vars]

    def get_model_variables(self):  # the data prototype in torch is Tensor
        return self.var_list
