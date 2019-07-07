from tensorflow.python.ops import variables
import numpy as np
from functools import reduce
import tensorflow as tf

from .var_man import BaseVarMan


class TFVariableManage(BaseVarMan):
    def __init__(self, sess):
        """
        implement of TensorFlow
        need session to modify or get values
        """
        super().__init__()
        # sess.run(tf.initialize_all_variables())
        self.sess = sess
        self.all_var = None
        self.var_list = None
        self.assign_ports = []
        self.assign_ops = []
        self.var_pos_dict = []  # (对应的变量下标, 对应变量中位置的下标)
        self.hash_dict = []
        self.var_num = 0

    def var_tolist(self, vars):
        """
        :param vars: [tf.Variable]
        :return: [numpy.array]
        """
        if self.var_list is None:
            self.var_list = vars
            self.build_key_var_hash()
        return self.sess.run(vars)

    def pull_val(self, val_idx):
        self.sess.run([self.assign_ops[k] for k in val_idx],
                      feed_dict={self.assign_ports[k]: self.val_list[k] for k in val_idx})

    def pull(self, keys=None):
        if keys is None:
            self.sess.run(self.assign_ops,
                          feed_dict={assign: val for assign, val in zip(self.assign_ports, self.val_list)})
        else:
            self.sess.run([self.assign_ops[k] for k in keys],
                          feed_dict={self.assign_ports[k]: self.val_list[k] for k in keys})

    def build_key_var_hash(self):
        f = lambda x, y: x * y
        self.hash_dict = []
        self.var_num = 0
        for i in range(len(self.var_list)):
            var = self.var_list[i]
            sp = var.shape.as_list()
            port = tf.placeholder(shape=sp, dtype=var.dtype)
            self.assign_ports.append(port)
            self.assign_ops.append(tf.assign(var, port))
            size = reduce(f, sp)
            #  size = var.size
            self.var_num += size
            for idx in range(size):
                self.var_pos_dict.append((i, idx))

    def refresh(self):
        """
        get newest variables
        :return:
        """
        self.var_list = variables.trainable_variables()
        return self.get_all_variables()

    def get_model_variables(self):
        if self.var_list is None:
            self.var_list = variables.trainable_variables()
            self.build_key_var_hash()

        return self.var_list

    def get_all_variables(self):
        var = []
        for v in self.var_list:
            var.append(self.sess.run(v))
        self.all_var = var
        return var

    def get_list_variables(self, var_list=None):
        """
        trans variables to series
        :param var_list: list(np.array)
        :return: np.array with shape [None]
        """
        if var_list is None:
            if self.all_var is None:
                self.get_all_variables()
            var_list = self.all_var
        lst = np.array([])
        for v in var_list:
            lst = np.concatenate([lst, v.flatten()])
        return lst

    def assign_key_val(self, vdict):
        """
        assign value to variables
        :param vdict: {index:value}
        :return:
        """
        if self.all_var is None:
            self.refresh()
        key_map = {}
        val_map = {}
        for k in vdict:
            v = vdict[k]
            k = int(k)
            idx, pos = self.var_pos_dict[k]
            try:
                key_map[idx].append(pos)
                val_map[idx].append(v)
            except:
                key_map[idx] = [pos]
                val_map[idx] = [v]
        feed_dict = {}
        ops = []
        for idx in key_map:
            flatten_var = self.all_var[idx].flatten()
            #  print(flatten_var[key_map[idx]], val_map[idx])
            flatten_var[key_map[idx]] = val_map[idx]
            self.all_var[idx] = flatten_var.reshape(self.all_var[idx].shape)
            feed_dict[self.assign_ports[idx]] = self.all_var[idx]
            ops.append(self.assign_ops[idx])
        self.sess.run(ops, feed_dict=feed_dict)
