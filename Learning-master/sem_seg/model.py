import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
from tf_grouping import group_point, knn_point

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32,
                                     shape=(batch_size, num_point, 9))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_point))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx3 gray image """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value

    input_image = tf.expand_dims(point_cloud, -1)
    
    #xyz_withnor in SpiderCNN is the point_cloud here
    K_knn = 16
    taylor_channel = 3

    xyz = point_cloud[:, :, 0:3]

    # SpiderCNN CONV
    with tf.variable_scope('delta') as sc:
        _, idx = knn_point(K_knn, xyz, xyz)
        
        grouped_xyz = group_point(xyz, idx)   
        point_cloud_tile = tf.expand_dims(xyz, [2])
        point_cloud_tile = tf.tile(point_cloud_tile, [1, 1, K_knn, 1])
        delta = grouped_xyz - point_cloud_tile
    
    with tf.variable_scope('SpiderConv1') as sc:
        feat_1 = tf_util.spiderConv(point_cloud, idx, delta, 16, taylor_channel = taylor_channel, 
                                        bn=True, is_training=is_training, bn_decay=bn_decay)

    with tf.variable_scope('SpiderConv2') as sc:
        feat_2 = tf_util.spiderConv(feat_1, idx, delta, 32, taylor_channel = taylor_channel, 
                                        bn=True, is_training=is_training, bn_decay=bn_decay)

    with tf.variable_scope('SpiderConv3') as sc:
        feat_3 = tf_util.spiderConv(feat_2, idx, delta, 64, taylor_channel = taylor_channel, 
                                        bn=True, is_training=is_training, bn_decay=bn_decay)
    point_feat = tf.concat([feat_1, feat_2, feat_3], 2)
    global_fea = tf_util.topk_pool(point_feat, k = 2, scope='topk_pool')
    global_fea = tf.reshape(global_fea, [batch_size, -1])

    global_fea = tf.expand_dims(global_fea, 1)
    global_fea = tf.expand_dims(global_fea, 1)
    global_fea_expand = tf.tile(global_fea, [1, num_point, 1, 1])

    # CONV
    net = tf_util.conv2d(input_image, 64, [1,9], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    
    
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    
    
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    
    points_feat1 = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    #points_feat1 = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
    #                     bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)


    # MAX
    pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point,1], padding='VALID', scope='maxpool1')

    # FC
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    #print(pc_feat1)

    # CONCAT 
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_point, 1, 1])
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand, global_fea_expand])

    print("the points_feat1_concat is", points_feat1_concat)
    try:
        # CONV 
        net = tf_util.conv2d(points_feat1_concat, 512, [1,1], padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training, scope='conv6')
        net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                             bn=True, is_training=is_training, scope='conv7')
        net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
        net = tf_util.conv2d(net, 13, [1,1], padding='VALID', stride=[1,1],
                             activation_fn=None, scope='conv8')
        net = tf.squeeze(net, [2])
        print(net)
    except:
        print("error occured")

    return net

def get_loss(pred, label):
    """ pred: B,N,13
        label: B,N """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    return tf.reduce_mean(loss)

if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32,4096,9))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a:np.random.rand(32,4096,9)})
            print(time.time() - start)
