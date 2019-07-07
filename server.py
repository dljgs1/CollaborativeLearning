# -*- coding:UTF-8 -*-
"""
协同学习的参数服务器 （通用接口版）

    hw 2019 1 17
"""

from flask import *
import json
import random
from myutils.format import unpack_rcvdata, float2bytes, unpack_extrainfo
import collections

app = Flask(__name__)
param_pool = {}
grad_que = []  # 梯度缓冲池，用于页面展示
grad_range = [-1.0, 1.0]  # 梯度范围

# 主页
@app.route('/')
def index():
    return render_template('index.html')


# 参数查看页面 ：
@app.route('/parameters', methods=['POST', 'GET'])
def param():
    if request.method == 'POST':  # 拿走最新的梯度
        if len(grad_que) > 0:
            ret = json.dumps(grad_que)
            grad_que.clear()
            return ret
        else:
            return "{}"
    return render_template('parameters.html', param=param_pool)


# 只有上传了梯度才会回传参数 否则一律不管 反正谁也不管谁的初始化是多少
def get_parameter(grad: dict, info=None) -> dict:
    global param_pool
    L = len(grad)  # L is |download|

    print("receive g:", str(grad)[:150])

    grad_que.append({"gradient": grad, "node_id": info["node_id"]})  # 保存当前的参数
    for p in grad:
        if p not in param_pool:
            param_pool[p] = grad[p]
        else:
            param_pool[p] += grad[p]
    keys = list(param_pool.keys())
    random.shuffle(keys)
    keys = keys[:L]  #
    values = [float2bytes(param_pool[k]) for k in keys]  # 把回传序列化 保证float的高效传输
    return {"keys": keys, "values": values}


# 参数更新接口 使用POST上传梯度 PS会返回一定量的参数 （返回多少取决于心情？）
# 参数上传和返回格式均为 key-value 模型一经确定 key值均确定
# TODO 为了减少上传量，可以采取稀疏编码，之后再讨论，先做出原型
@app.route('/update', methods=['POST'])
def update():
    if request.method == 'POST':
        content = request.form.get('gradient')
        if content is None or content == "":
            return "{'msg':'error'}"
        print("receive:", str(content)[:150])
        ret = get_parameter(unpack_rcvdata(content), unpack_extrainfo(content))  # 这里可以添加额外信息
        return json.dumps(ret)


class ParameterServer:
    def __init__(self, ip, port):
        self.ip = ip
        self.port = int(port)

    def start_service(self):  # 一个进程只能跑一个（吧？） 所以这可以是一个单机多进程实验
        app.run(host=self.ip, port=self.port, debug=True)


"""
使用方法：
from server import ParameterServer
server1 = ParameterServer("xxx:xxx:xxx",5000)
"""

if __name__ == '__main__':
    # server1 = ParameterServer("192.168.124.65", 5000)
    server1 = ParameterServer("0.0.0.0", 5000)
    # server1 = ParameterServer("::", 5000)
    server1.start_service()
