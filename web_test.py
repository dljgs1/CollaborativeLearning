"""


web主线程 设计完了之后只进行热更新 这个代码不再重写

"""

from flask import *
import importlib as imp


app = Flask(__name__)  # 创建1个Flask实例


@app.route('/', methods=['POST', 'GET'])  # 路由系统生成 视图对应url,1. decorator=app.route() 2. decorator(first_flask)
def index():  # 视图函数
    if request.method == 'GET':
        return render_template('index.html')
    else:
        fname = request.form.get('fname')
        content = request.form.get('fcontent')
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  # 启动socket
