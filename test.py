import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float(
    'learn_rate', 0.001, 'learning rate 0. <1.0')
tf.app.flags.DEFINE_integer(
    'dst_label', 0, 'input a int between ')
tf.app.flags.DEFINE_integer(
    'port', 5000, 'port of client')

FLAGS = tf.app.flags.FLAGS
learning_rate = FLAGS.learn_rate
dst_label = FLAGS.dst_label
port = FLAGS.port
ip = "192.168.124.65"


def compute_accuracy(v_x, v_y):
    global prediction
    # input v_x to nn and get the result with y_pre
    y_pre = sess.run(prediction, feed_dict={x: v_x, keep_prob: 1})
    # find how many right
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    # calculate average
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # get input content
    result = sess.run(accuracy, feed_dict={x: v_x, y: v_y, keep_prob: 1})
    return result


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


#  ============================================================================

def conv2d(x, W):
    # strides=[1,x_movement,y_movement,1]
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# CNN
def _add_layer(inputs, in_size, out_size, keep_prob):
    # init w: a matric in x*y
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # init b: a matric in 1*y
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    # calculate the result
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # add the active hanshu
    # reshape(data you want to reshape, [-1, reshape_height, reshape_weight, imagine layers]) image layers=1 when the imagine is in white and black, =3 when the imagine is RGB
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # ********************** conv1 *********************************
    # transfer a 5*5*1 imagine into 32 sequence
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    # input a imagine and make a 5*5*1 to 32 with stride=1*1, and activate with relu
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # output size 28*28*32
    h_pool1 = max_pool_2x2(h_conv1)  # output size 14*14*32

    # ********************** conv2 *********************************
    # transfer a 5*5*32 imagine into 64 sequence
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    # input a imagine and make a 5*5*32 to 64 with stride=1*1, and activate with relu
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # output size 14*14*64
    h_pool2 = max_pool_2x2(h_conv2)  # output size 7*7*64

    # ********************* func1 layer *********************************
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    # reshape the image from 7,7,64 into a flat (7*7*64)
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # ********************* func2 layer *********************************
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return prediction


# MLP
def add_layer(inputs, in_size, out_size, activation_function=None, ):
    # init w: a matric in x*y
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # init b: a matric in 1*y
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    # calculate the result
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # add the active hanshu
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    return outputs


# filt labels in dataset ipt except label dst
def filt_datalabels(ipt, dst, one_hot=True):
    if one_hot:
        idx = ipt.train._labels[:, dst] > 0.9
    else:
        idx = ipt.train._labels == dst
    ipt.train._images = ipt.train._images[idx]
    ipt.train._labels = ipt.train._labels[idx]
    ipt.train._num_examples = len(ipt.train._labels)


# load mnist data
mnist = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
# filt_datalabels(mnist, dst_label)

# define placeholder for input
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
# add layer
prediction = add_layer(x, 784, 10, activation_function=tf.nn.softmax)# , keep_prob)
# calculate the loss
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(y * tf.log(tf.clip_by_value(prediction, 1e-10, 10.0)), reduction_indices=[1]))
# use Gradientdescentoptimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
config.gpu_options.allow_growth = True
# init session
sess = tf.Session(config=config)
# init all variables
sess.run(tf.global_variables_initializer())

# sess.run(tf.random_normal_initializer())
# start training

from tensorflow.python.ops import variables

# print("vec:", sess.run(variables.trainable_variables()))

from client import ParameterClient
from myutils.tftools import TFVariableManage

var_man = TFVariableManage(sess)
client1 = ParameterClient(ip, port, var_man, node_id=dst_label)
client1.before_train(lambda x: print("初始化参数：", x))


def train(data_flow, train_op, step_num=1000):
    for i in range(step_num):
        # get batch to learn easily
        batch_x, batch_y = data_flow.train.next_batch(100)
        sess.run(train_op, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        if i % 50 == 0:
            print(i, "更新参数")
            client1.update()  # 更新参数
            print(compute_accuracy(data_flow.test.images, data_flow.test.labels))


# experiment 1 :
# gradient hot map on different labels of datasets
train(mnist, train_step, step_num=1000)

# experiment 2 : posion投毒
