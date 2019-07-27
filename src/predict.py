# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     predict
   Author :        Xiaosong Zhou
   date：          2019/7/23
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

import pandas as pd
import numpy as np
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)
from src.model_1 import RecommenderNetworkConfig, RecommenderNetwork
# 预测函数

USER_DIR = '../mid_data/user_feature0.01.csv'
ITEM_DIR = '../mid_data/item_feature0.01.csv'

def rating_item(network, user_id, item_id):
    # 预测单个用户对单个商品的评分（映射过来就是购买行为）
    res_user = np.reshape(user_pd[user_pd['user_id'] == user_id].values, [2]).tolist()
    res_item = np.reshape(item_pd[item_pd['item_id'] == item_id].values, [2]).tolist()
    user_id = int(res_user[0])
    user_vectors = res_user[1]
    user_vectors = list(map(int, user_vectors[1:-1].split(',')))
    user_vectors = np.reshape(np.array(user_vectors), [1, config.user_dim])
    item_id = int(res_item[0])
    item_vectors = res_item[1]
    item_vectors = list(map(int, item_vectors[1:-1].split(',')))
    item_vectors = np.reshape(np.array(item_vectors), [1, config.item_dim])

    inference = network.model([user_vectors.astype('float32'), item_vectors.astype('float32')])
    a = inference.numpy()
    return a[0][0]






if __name__ == '__main__':
    user_pd = pd.read_csv(USER_DIR,sep=',', header=None, names=['user_id', 'user_vector'])
    item_pd = pd.read_csv(ITEM_DIR,sep=',', header=None, names=['item_id', 'item_vector'])
    config = RecommenderNetworkConfig()
    network = RecommenderNetwork(config)
    result = rating_item(network, user_id='3063489302', item_id='878909776')
    print(result)
