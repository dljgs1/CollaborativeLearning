from .var_man import BaseVarMan

import keras


class KerasVariableManage(BaseVarMan):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.var_list = self.model.trainable_weights

    def pull_val(self, val_idx: list):
        keras.backend.batch_set_value([(self.var_list[i], self.val_list[i]) for i in val_idx])

    def var_tolist(self, vars=None):
        if vars is None:
            return self.model.get_weights()
        else:
            return keras.backend.batch_get_value(vars)

    def get_model_variables(self):
        if self.var_list is None:
            self.var_list = self.model.trainable_weights
        return self.var_list
