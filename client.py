# -*- coding:UTF-8 -*-
"""
CoL客户端
    要对各个框架版本进行适配，现在暂时只做TensorFlow的
    hw 2019 - 1 - 17

"""

import requests
import random
import numpy as np

from myutils.tftools import *
from myutils.kerastools import *
from myutils.torchtools import *
from myutils.format import pack_senddata, unpack_rcvdata

adaption_dict = {
    "TensorFlow": TFVariableManage,  # TEST DONE
    "Keras": KerasVariableManage,  # TODO:TEST
    "PyTorch": TorchVariableManage,  # TODO:TEST
    "Theano": None,  # TODO
    "test": None
}

# 适配 —— 有没有可能跨框架训练？ ——不太可能

# 训练前后调用这两函数 用以上传梯度 频率可以控制 可以是固定的batch上传 可以是epoch上传
import json
import copy


class ParameterClient:
    """
    Client used by data holder
    Need : python3.6 / tensorflow / kears / pytorch / theano
    """

    def __init__(self, ip, port, model: BaseVarMan, node_id=None):  # 后续可以加认证之类的 先免了
        self.ip = ip
        self.port = port
        self.model = model
        self.node_id = node_id
        self.param = None
        self.grad = None
        self.receive_dict = {}

    @classmethod
    def register_model(cls, framework, args):
        cls.__framework__ = framework
        try:
            return adaption_dict[framework](args)
        except KeyError:
            print("No such framework -> %s " % (framework,))
        except TypeError:
            print("Error in args -> %s , check your args" % (str(args),))

    def before_train(self, callback=None):
        self.model.push(self.model.get_model_variables())
        self.param = copy.deepcopy(self.model.vec)
        if callback is not None:
            return callback(self.param)

    def update(self, selectfun=None, callback=None):
        self.model.push(self.model.get_model_variables())
        param = copy.deepcopy(self.model.vec)
        self.grad = param - self.param
        self.param = param
        print("grad:", str(self.grad)[:50])
        print("param:", str(self.param)[:50])
        if selectfun is not None:
            send = selectfun(self.grad)  # you need to implement function selectfun to select gradients to update
        else:  # 最简单策略 选择阈值裁剪上传 最大20%
            idx = [i for i in range(len(self.grad))]
            idx.sort(key=lambda x: abs(self.grad[x]), reverse=True)
            # idx = [i for i in range(len(self.grad)) if abs(self.grad[i]) >= 0.00001]
            # random.shuffle(idx)
            # idx = idx[:int(len(idx) / 5)]
            send = {i: max(min(1.0, self.grad[i]), -1.0) for i in idx[:int(len(idx) / 10)]}

        # 在此处添加额外信息
        if self.node_id is not None:
            send["node_id"] = self.node_id

        sendstr = pack_senddata(send)
        print("send (%d):" % len(send), sendstr[:50])
        r = requests.post("http://" + self.ip + ":" + str(self.port) + "/update", {"gradient": sendstr})
        self.receive_dict = unpack_rcvdata(r.text)
        print("receive (%d):" % len(self.receive_dict), str(self.receive_dict)[:150])
        if callback is not None:  # use callback , you need to do some work on receive_dict and apply it to model
            callback(self)
        else:
            self.model.increase_modify_var(self.receive_dict)  # default: update all the parameters
        print(self.model.vec)

        self.model.pull()  # update

        # self.param = r# ? 不对 回传的是部分参数 需要选择位置进行更新 所以参数必须要能转成一维 也能往回转 否则就又问题


class HTTPClient:
    """
    client as a server on worker, receiving data and modle from data provider and model provider
    need ： python3
    """

    def __init__(self, address):
        pass


if __name__ == '__main__':  # 以服务的方式运行客户机——作为一个可分配任务的worker
    pass
