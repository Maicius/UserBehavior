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
import json
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


def get_user_item_list(user_dir, item_dir, json_dirs):
    user_pd = pd.read_csv(user_dir,sep=',', header=None, names=['user_id', 'gender', 'age',
                                                                'career', 'income', 'stage'],
                          low_memory=False)
    item_pd = pd.read_csv(item_dir,sep=',', header=None, names=['item_id', 'cate_1_id', 'cate_id',
                                                                'brand_id', 'price'],
                          low_memory=False)
    num_user_unfind = 0
    num_item_unfind = 0
    user_pd = user_pd.groupby(by=['user_id'])
    item_pd = item_pd.groupby(by=['item_id'])
    # 直接用数值保存下来
    user_iterator = 0
    write_file = open('../mid_data_random/predict_list','w',encoding='utf-8')
    for json_dir in json_dirs:
        with open(json_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for sub_data in data:
                # 每一个用户
                user_id = sub_data['user_id']
                user_iterator += 1
                # res_user_pd = user_pd[user_pd['user_id'].isin([str(user_id)])].values
                try:
                    res_user_pd = user_pd.get_group(str(user_id))
                except KeyError:
                    num_user_unfind += 1
                    continue
                res_user = np.reshape(
                    res_user_pd.values, [6]
                ).tolist()

                # 一系列item
                item_list = str(sub_data['item_list']).strip().split(',')
                if len(item_list) <= 50:
                    continue
                # res_items_pd = item_pd[item_pd['item_id'].isin(item_list)].values
                res_items_pd = []
                for item_id in item_list:
                    try:
                        df_tmp = item_pd.get_group(str(item_id))
                    except KeyError:
                        num_item_unfind += 1
                        continue
                    res_items_pd.append(df_tmp.values)

                res_items = np.reshape(
                    res_items_pd, [len(res_items_pd), 5]
                ).tolist()
                # 写入
                print('写入用户：'+str(user_iterator))
                write_file.write(' '.join(res_user))
                write_file.write('||')
                for item in res_items:
                    write_file.write('||')
                    write_file.write(' '.join(item))
                write_file.write('\n')
    write_file.close()

    print(num_user_unfind)
    print(num_item_unfind)


def predict(network, user_vector_list, item_vector_list):
    assert (len(user_vector_list) == len(item_vector_list))
    test_batch_size = len(user_vector_list)

    # user_id,gender,age,career,income,stage
    # user = np.reshape(user_vector_list, [test_batch_size, 6])
    user = np.asarray(user_list)

    # item_id,cate_1_id,cate_id,brand_id,price
    # item = np.reshape(item_vector_list, [test_batch_size, 5])
    item = np.asarray(item_list)

    inference = network.model([np.reshape(user[:, 1], [test_batch_size, 1]).astype(np.float32),
                               np.reshape(user[:, 2], [test_batch_size, 1]).astype(np.float32),
                               np.reshape(user[:, 3], [test_batch_size, 1]).astype(np.float32),
                               np.reshape(user[:, 4], [test_batch_size, 1]).astype(np.float32),
                               np.reshape(np.asarray([np.asarray(x) for x in user[:, 5]]), [test_batch_size, 6]).astype(np.float32),
                               np.reshape(item[:, 1], [test_batch_size, 1]).astype(np.float32),
                               np.reshape(item[:, 2], [test_batch_size, 1]).astype(np.float32),
                               np.reshape(item[:, 3], [test_batch_size, 1]).astype(np.float32),
                               np.reshape(item[:, 4], [test_batch_size, 1]).astype(np.float32)],
                              training=False)

    pred = inference.numpy()
    return pred


if __name__ == '__main__':
    config = RecommenderNetworkConfig()
    network = RecommenderNetwork(config)
    # result = rating_item(network, user_id='452224162', item_id='250302368')
    #
    # print(result)

    # get_user_item_list(user_dir='../mid_data_random/user_feature_vector_Random.csv',
    #                    item_dir='../mid_data_random/item_feature_vector_Random.csv',
    #                    json_dirs=['../mid_data/tmp_result0.json',
    #                               '../mid_data/tmp_result1.json',
    #                               '../mid_data/tmp_result2.json',
    #                               '../mid_data/tmp_result3.json',
    #                               '../mid_data/tmp_result4.json'])

    user_list = [[10000100, 0, 7, 1, 8, [1, 6, 4, 2, 0, 0]],
                 [10000100, 0, 7, 1, 8, [1, 6, 4, 2, 0, 0]],
                 [10000100, 0, 7, 1, 8, [1, 6, 4, 2, 0, 0]],
                 [10000100, 0, 7, 1, 8, [1, 6, 4, 2, 0, 0]]]
    item_list = [[463911171, 61, 2151, 13395, 2.0],
                 [900775433, 61, 2151, 13395, 1.0],
                 [1048597011, 92, 1982, 104059, 9.0],
                 [953063189, 107, 452, 80840, 18.0]]
    pred = predict(network, user_list, item_list)

