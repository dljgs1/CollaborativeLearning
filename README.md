# 框架无关的顶层协同学习环境

## 当前框架支持
 - TensorFlow
 - Keras
 - Pytorch
 
## 通信设置
 - 协议：HTTP
 - 结构： S2C
 
 
## 项目目录
（待补充）

## 运行/使用说明
- 参数服务器： python server.py | 默认工作在5000端口
- 工作机： python worker.py | 工作在5001端口，目前使用服务器模式，后续会改为客户端模式。
- 在工作机运行test.py 将模型加载
## 拓展
修改test.py为自己的模型，当前的测试模型为DNN训练MNIST


 

 
