# Copyright (C) 2019 Chao Wen, Yinda Zhang, Zhuwen Li, Yanwei Fu
# All rights reserved.
# This code is licensed under BSD 3-Clause License.
import tensorflow as tf
import tflearn
import numpy as np
import pprint
import pickle
import shutil
import os
import json

from modules.models_mvp2m import MeshNetMVP2M as MVP2MNet
from modules.models_p2mpp import MeshNet as P2MPPNet
from modules.config import execute
# from utils.dataloader import DataFetcher
from utils.tools import construct_feed_dict, load_demo_image
# from utils.visualize import plot_scatter

from sklearn.preprocessing import MinMaxScaler

def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    # ---------------------------------------------------------------
    # Set random seed
    print('=> pre-porcessing')
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    # ---------------------------------------------------------------
    num_blocks = 3
    num_supports = 2
    placeholders = {
        'features': tf.placeholder(tf.float32, shape=(None, 3), name='features'),
        'img_inp': tf.placeholder(tf.float32, shape=(3, 224, 224, 3), name='img_inp'),
        'labels': tf.placeholder(tf.float32, shape=(None, 6), name='labels'),
        'support1': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support2': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'support3': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'faces': [tf.placeholder(tf.int32, shape=(None, 4)) for _ in range(num_blocks)],
        'edges': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks)],
        'lape_idx': [tf.placeholder(tf.int32, shape=(None, 10)) for _ in range(num_blocks)],  # for laplace term
        'pool_idx': [tf.placeholder(tf.int32, shape=(None, 2)) for _ in range(num_blocks - 1)],  # for unpooling
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32),
        'sample_coord': tf.placeholder(tf.float32, shape=(43, 3), name='sample_coord'),
        'cameras': tf.placeholder(tf.float32, shape=(3, 5), name='Cameras'),
        'faces_triangle': [tf.placeholder(tf.int32, shape=(None, 3)) for _ in range(num_blocks)],
        'sample_adj': [tf.placeholder(tf.float32, shape=(43, 43)) for _ in range(num_supports)],
    }

    # step = cfg.test_epoch
    # root_dir = os.path.join(cfg.save_path, cfg.name)
    model1_dir = os.path.join('results', 'coarse_mvp2m', 'models')
    model2_dir = os.path.join('results', 'refine_p2mpp', 'models')
    # predict_dir = os.path.join(cfg.save_path, cfg.name, 'predict', str(step))
    # if not os.path.exists(predict_dir):
    #     os.makedirs(predict_dir)
    #     print('==> make predict_dir {}'.format(predict_dir))
    # -------------------------------------------------------------------
    print('=> build model')
    # Define model
    model1 = MVP2MNet(placeholders, logging=True, args=cfg)
    model2 = P2MPPNet(placeholders, logging=True, args=cfg)
    # ---------------------------------------------------------------
    print('=> load data')
    demo_img_list = ['data/demo/item1.png',
                     'data/demo/item2.png',
                     'data/demo/item3.png']
    img_all_view = load_demo_image(demo_img_list)
    cameras = np.loadtxt('data/demo/cameras.txt')
    # data = DataFetcher(file_list=cfg.test_file_path, data_root=cfg.test_data_path, image_root=cfg.test_image_path, is_val=True)
    # data.setDaemon(True)
    # data.start()
    # ---------------------------------------------------------------
    print('=> initialize session')
    sesscfg = tf.ConfigProto()
    sesscfg.gpu_options.allow_growth = True
    sesscfg.allow_soft_placement = True
    sess = tf.Session(config=sesscfg)
    sess.run(tf.global_variables_initializer())
    # sess2 = tf.Session(config=sesscfg)
    # sess2.run(tf.global_variables_initializer())
    # ---------------------------------------------------------------
    model1.load(sess=sess, ckpt_path=model1_dir, step=50)
    model2.load(sess=sess, ckpt_path=model2_dir, step=10)
    # exit(0)
    # ---------------------------------------------------------------
    # Load init ellipsoid and info about vertices and edges
    pkl = pickle.load(open('data/iccv_p2mpp.dat', 'rb'))
    # Construct Feed dict
    feed_dict = construct_feed_dict(pkl, placeholders)
    # ---------------------------------------------------------------
    tflearn.is_training(False, sess)
    print('=> start test stage 1')
    feed_dict.update({placeholders['img_inp']: img_all_view})
    feed_dict.update({placeholders['labels']: np.zeros([10, 6])})
    feed_dict.update({placeholders['cameras']: cameras})
    stage1_out3 = sess.run(model1.output3, feed_dict=feed_dict)

    print('stage1out:', stage1_out3.shape)

    v_min = np.array([stage1_out3[:, 0].min(), stage1_out3[:, 1].min(), stage1_out3[:, 2].min()])
    v_max = np.array([stage1_out3[:, 0].max(), stage1_out3[:, 1].max(), stage1_out3[:, 2].max()])
    print(v_min, v_max)

    print('=> loading features')
    # (3, 2466, 3)
    loaded_features = np.loadtxt('features.txt')
    index = 0
    print('loaded features:', loaded_features.shape)
    v = loaded_features

    v_old_min = np.array([loaded_features[:, 0].min(), loaded_features[:, 1].min(), loaded_features[:, 2].min()])
    v_old_max = np.array([loaded_features[:, 0].max(), loaded_features[:, 1].max(), loaded_features[:, 2].max()])

    in_feature = ((v - v_old_min) * (v_max - v_min) / (v_old_max - v_old_min)) + v_min

    # scaler = MinMaxScaler((v_min, v_max))
    # in_feature = scaler.fit_transform(loaded_features[0])
    print(in_feature.shape)
    # in_feature = ((v - v.min()) * (v_max - v_min) / (v.max() - v.min())) + v_min

    print('=> start test stage 2')
    feed_dict.update({placeholders['features']: in_feature})
    pred_vert, img_feat = sess.run([model2.output2l, model2.img_feat], feed_dict=feed_dict)

    pred_mesh = np.hstack((np.full([pred_vert.shape[0],1], 'v'), pred_vert))

    pred_path = 'data/demo/new_prediction.obj'
    np.savetxt(pred_path, pred_mesh, fmt='%s', delimiter=' ')
    print('=> save to {}'.format(pred_path))

    # v = in_feature
    # v_new_min = np.array([v[:, 0].min(), v[:, 1].min(), v[:, 2].min()])
    # v_new_max = np.array([v[:, 0].max(), v[:, 1].max(), v[:, 2].max()])
    # rescaled_back = in_feature = ((v - v_new_min) * (v_old_max - v_old_min) / (v_new_max - v_new_min)) + v_old_min

    scaled_mesh = np.hstack((np.full([in_feature.shape[0],1], 'v'), in_feature))

    pred_path = 'data/demo/rescaled_original.obj'
    np.savetxt(pred_path, scaled_mesh, fmt='%s', delimiter=' ')

    # img_feat_json = {}
    # classes = ['x0', 'x1', 'x2']

    # for i in range(len(placeholders['img_feat'])):
    #   img_feat_json[classes[i]] = str(img_feat[i])

    # with open('data/demo/img_feat.json', 'w') as f:
    #   json.dump(img_feat_json, f)

    print('=> save to {}'.format(pred_path))

if __name__ == '__main__':
    print('=> set config')
    args=execute()
    # pprint.pprint(vars(args))
    main(args)
