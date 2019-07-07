"""
worker:
    A distributed machine executing computation missions
    Receive code and data(with encryption)

    hw 2019 1 24
"""

from flask import *
import importlib as imp
import json

app = Flask(__name__)  # 创建1个Flask实例

workers_dict = {}


@app.route('/', methods=['POST', 'GET'])  # 路由系统生成 视图对应url,1. decorator=app.route() 2. decorator(first_flask)
def index():  # 视图函数
    if request.method == 'GET':
        return render_template('worker.html', workers=workers_dict)
    else:  # 接收任务类型
        print(request.form)
        msgtype = request.form.get('type')
        data = request.form.get('content')
        try:
            data = json.loads(data)
        except:
            return "error in decoding!"

        if msgtype == "register":  # 注册
            already = []
            for nid in data:
                if nid in workers_dict:
                    already.append(nid)
                workers_dict[nid] = data[nid]
            if len(already) > 0:
                return "替换的ID：" + "|".join(already)
            else:
                return "success"
        if msgtype == "mission":  # 分配任务
            pass

        return render_template('worker.html', workers=workers_dict)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)  # 本地服务的端口 5001
