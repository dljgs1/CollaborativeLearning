"""
变量通用管理 省的在client写一大堆
  hw 19.1.18

  TODO: 去掉字典使用 改为效率更高的array
"""

# 注意 var是变量 是一个matrix 或者scalar 或者array
# 存进来的时候 是一个list 用下标确定变量
# 存的可能是原型 也可能是数据 继承的时候自己定吧

import numpy as np


#
# 两个核心功能： 1 变量存进来 存成字典 2 指定字典 存回原型
# 两个需要实现的接口：1 变量->字典存储 2 值回写
# 可以重复的内容：var摊开为vec
#
class BaseVarMan:
    """
    functions need to be implement:
        var_tolist(vars: list) : transfer prototype to array
            vars:: data prototype list
            :return list of numpy array

        pull_val(val_idx) : write array to prototype tensor
            val_idx::  integer indexing the pos of prototype
            :return None

        get_model_parameters() : get prototype of weights
            :return list of Variables
    """

    def __init__(self):
        self.val_list = []  # 存储的各个数据的numpy array
        self.mesh_idx = None  # key:var_idx 面向val_list的映射
        self.mesh_range = None  # key:[idx_st,idx_ed]
        self.mesh = None  # key:val_idx 通常是1:1的 直接用self.vec[idx]获取即可
        self.vec = None  # 数据向量 是把所有变量摊平后的结果 这个是必须要的 vec的构建 最好是把摊平的各个向量迅速拼接

    # 在获取数据原型后建立的元信息——是否必要？
    def init_meta(self):
        key = 0
        self.mesh = {}
        self.mesh_idx = {}
        self.mesh_range = {}
        # print(self.val_list)
        for i in range(len(self.val_list)):
            var = self.val_list[i]
            sp = var.shape
            size = np.size(var)
            temp = var.flatten()
            self.mesh_range[i] = [key, key+size]
            for bias in range(size):
                k = key + bias
                self.mesh[k] = temp[bias]
                self.mesh_idx[k] = i
            key += size
        self.vec = np.zeros(key)
        print("vec size:", len(self.vec), key)
        print(self.mesh_range)

    # 获取一个字典 传输时使用
    def get_dict(self, keys):
        return {k: self.vec[int(k)] for k in keys}

    # 刷新vec值 获得最新val_list
    def flush_vec(self):
        for i in range(len(self.val_list)):
            v = self.val_list[i].flatten()
            st = self.mesh_range[i][0]
            ed = self.mesh_range[i][1]
            print(st, ed)
            self.vec[st:ed] = v

    # 将变量托管
    def push(self, vars):
        self.val_list = self.var_tolist(vars)
        if self.mesh_idx is None or self.mesh is None:  # 首次托管 需要存元信息
            self.init_meta()
        # 重新存储 是更新数值 只需要对vec进行更新即可 目的求快 因为此时数据已经到了list了 不应该再造字典
        self.flush_vec()

    # 把存储的变量值拉回来 通过keys获取 默认全部 —— 一个个赋值是不是太慢？可以有通用操作口？ 这里 可以默认传list
    def pull(self, keys=None):
        if keys is None:
            self.pull_val([i for i in range(len(self.val_list))])
        else:
            self.pull_val(list(set(self.mesh_idx[int(k)] for k in keys)))

    # 将vec的值写回 cope为需要覆写的变量范围
    def vec2val(self, scope):
        for s in scope:
            val = self.val_list[s]
            rg = self.mesh_range[s]
            sp = val.shape
            val[:] = np.reshape(self.vec[rg[0]:rg[1]], sp)

    # 将某些k修改为v 并且应用到val_list 上
    def modify_var(self, kv: dict):
        scope = set()
        for k in kv:
            self.vec[int(k)] = kv[k]
            scope.add(self.mesh_idx[k])
        self.vec2val(scope)

    # 增量修改
    def increase_modify_var(self, kv: dict):
        scope = set()
        for k in kv:
            self.vec[int(k)] += kv[k]
            scope.add(self.mesh_idx[k])
        self.vec2val(scope)

    # ============== need to implement ==============

    # 将原型转换为numpy
    def var_tolist(self, vars) -> list:
        raise NotImplementedError

    # 拉取某个变量值——存储
    def pull_val(self, val_idx: list):
        raise NotImplementedError

    # 返回参数原型
    def get_model_variables(self) -> list:
        raise NotImplementedError
