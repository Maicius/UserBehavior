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
from src.model_dynamic import RecommenderNetworkConfig, RecommenderNetwork
# 预测函数 动态列表输入形式

USER_DIR = '../mid_data/user_feature_vector_Random.csv'
ITEM_DIR = '../mid_data/item_feature_vector_Random.csv'


def rating_item(network, user_id, item_id):
    # 预测单个用户对单个商品的评分（映射过来就是购买行为）
    # user_id,gender,age,career,income,stage
    res_user = np.reshape(user_pd[user_pd['user_id'] == user_id].values, [6]).tolist()
    # item_id,cate_1_id,cate_id,brand_id,price
    res_item = np.reshape(item_pd[item_pd['item_id'] == item_id].values, [5]).tolist()
    # user
    user_id = int(res_user[0])
    gender = int(res_user[1])
    age = int(res_user[2])
    career = int(res_user[3])
    income = int(res_user[4])
    stage = res_user[5]
    stage = list(map(int, stage[1:-1].split(',')))

    #item
    item_id = int(res_item[0])
    cate_1_id = int(res_item[1])
    cate_id = int(res_item[2])
    brand_id = int(res_item[3])
    price = float(res_item[4])

    inference = network.model([np.reshape(gender, [1, 1]).astype(np.float32),
                               np.reshape(age, [1, 1]).astype(np.float32),
                               np.reshape(career, [1, 1]).astype(np.float32),
                               np.reshape(income, [1, 1]).astype(np.float32),
                               np.reshape(stage, [1, 6]).astype(np.float32),
                               np.reshape(cate_1_id, [1, 1]).astype(np.float32),
                               np.reshape(cate_id, [1, 1]).astype(np.float32),
                               np.reshape(brand_id, [1, 1]).astype(np.float32),
                               np.reshape(price, [1, 1]).astype(np.float32)],
                              training=False)

    a = inference.numpy()
    return a[0][0]


if __name__ == '__main__':

    user_pd = pd.read_csv(USER_DIR,sep=',', header=None, names=['user_id', 'gender', 'age',
                                                                'career', 'income', 'stage'])
    item_pd = pd.read_csv(ITEM_DIR,sep=',', header=None, names=['item_id', 'cate_1_id', 'cate_id',
                                                                'brand_id', 'price'])
    config = RecommenderNetworkConfig()
    network = RecommenderNetwork(config)
    # result = rating_item(network, user_id='64985365', item_id='852475635')
    # result = rating_item(network, user_id='66014521', item_id='1177855544')
    # result = rating_item(network, user_id='67137132', item_id='1079925753')
    # result = rating_item(network, user_id='68498107', item_id='1147963479')
    result = rating_item(network, user_id='452224162', item_id='250302368')

    print(result)
