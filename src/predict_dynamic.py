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
import multiprocessing

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)
from src.model_dynamic import RecommenderNetworkConfig, RecommenderNetwork
# 预测函数 动态列表输入形式
import datetime
USER_DIR = '../mid_data/user_feature_vector_Random.csv'
ITEM_DIR = '../mid_data/item_feature_vector_Random.csv'
mid_pre = '../mid_data/'
PROCESS_NUM = 7
res_pre = '../result_data/'
PRE = "../raw_data/ECommAI_ubp_round1_"
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


def change_id_2_vec(json_dir, user_pd, item_pd):
    file_dir = 'id_2_vec_' + json_dir.split('/')[-1]
    user_iterator = 0
    num_user_unfind = 0
    num_item_unfind = 0
    write_file = open('../mid_data_random/' + file_dir,'w',encoding='utf-8')
    less_than_50 = []
    with open(json_dir, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for sub_data in data:
            # 一系列item
            item_list = str(sub_data['item_list']).strip().split(',')
            if len(item_list) <= 50:
                less_than_50.append(sub_data)


            # # 每一个用户
            user_id = sub_data['user_id']
            user_iterator += 1
            if user_iterator % 1000 == 0:
                print(file_dir, user_iterator, datetime.datetime.now())
            #
            # res_user_pd = user_pd[user_pd['user_id'].isin([str(user_id)])].values
            try:
                res_user_pd = user_pd.get_group(str(user_id))
            except KeyError:
                num_user_unfind += 1
                continue
            res_user = np.reshape(
                res_user_pd.values, [6]
            ).tolist()


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

            write_file.write(' '.join(res_user))
            write_file.write('||')
            for item in res_items:
                write_file.write('||')
                write_file.write(' '.join(item))
            write_file.write('\n')
    with open("../mid_data/_less" + file_dir, 'w', encoding='utf-8') as w:
        json.dump(less_than_50, w)
    print(num_user_unfind)
    print(user_iterator)
    print(num_item_unfind)

def get_user_item_list(user_dir, item_dir, json_dirs):
    user_pd = pd.read_csv(user_dir,sep=',', header=None, names=['user_id', 'gender', 'age',
                                                                'career', 'income', 'stage'],
                          low_memory=False)
    item_pd = pd.read_csv(item_dir,sep=',', header=None, names=['item_id', 'cate_1_id', 'cate_id',
                                                                'brand_id', 'price'],
                          low_memory=False)

    user_pd = user_pd.groupby(by=['user_id'])
    item_pd = item_pd.groupby(by=['item_id'])
    # 直接用数值保存下来

    process_list = []
    for json_dir in json_dirs:
        p = multiprocessing.Process(target=change_id_2_vec,
                                    args=[json_dir, user_pd, item_pd])
        p.daemon = True
        process_list.append(p)
    for p in process_list:
        p.start()

    for p in process_list:
        p.join()
    print("finish")
    result = []
    all_user_dict = []
    for i in range(5):
        with open(mid_pre + "_lessid_2_vec_tmp_result" + str(i) + '.json', 'r', encoding='utf-8') as r:
            data = json.load(r)
            all_user_dict.extend(data)
    with open(mid_pre + "_less_id_2_vec_result.json", "w", encoding='utf-8') as w:
        json.dump(all_user_dict, w)

    for json_dir in json_dirs:
        file_dir = '../mid_data_random/' + 'id_2_vec_' + json_dir.split('/')[-1]
        with open(file_dir, 'r', encoding='utf-8') as r:
            data = r.readlines()
        result.extend(data)
    with open('../mid_data_random/predict_list', 'w', encoding='utf-8') as w:
        w.writelines(result)



def predict(network, user_vector_list, item_vector_list):
    assert (len(user_vector_list) == len(item_vector_list))
    test_batch_size = len(user_vector_list)

    # user_id,gender,age,career,income,stage
    # user = np.reshape(user_vector_list, [test_batch_size, 6])
    user = np.asarray(user_vector_list)

    # item_id,cate_1_id,cate_id,brand_id,price
    # item = np.reshape(item_vector_list, [test_batch_size, 5])
    item = np.asarray(item_vector_list)

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
    item_vector_df = pd.DataFrame(item_vector_list)
    item_vector_df['score'] = pred
    top50 = item_vector_df.sort_values(by='score', ascending=False)[0].head(50).values

    return top50

x = '100216777 0 7 8 3 [7, 0, 0, 0, 0, 0]'
def change_user_str_2_vec(x):
    arr = x[:-1].split('[')
    feature = list(map(int, arr[0].strip().split(' ')))
    state = list(map(int, arr[1].split(', ')))
    feature.append(state)
    return feature

def change_item_feature_str_2_vec(x):
    arr = x.split('||')
    data = []
    for item in arr:
        item = item[:-2]
        temp = list(map(int, item.split(' ')))
        temp[-1] = float(temp[-1])
        data.append(temp)
    return data

def do_predict(index, data_df):
    config = RecommenderNetworkConfig()
    network = RecommenderNetwork(config)
    count = 0
    result = []
    for data in data_df.values:
        if count % 1000 == 0:
            print(index, count, datetime.datetime.now())
        item_list = data[1]
        num = len(item_list)
        # 一次传入一个用户
        user_list = [data[0]] * num
        top50 = predict(network, user_list, item_list)
        predict_list = map(str, list(top50))
        predict_list = ",".join(predict_list)
        result.append(dict(user_id=int(data[0][0]), item_list=predict_list))
        count += 1

    with open(mid_pre + "_pred_tmp_result" + str(index) + ".json", 'w', encoding='utf-8') as w:
        json.dump(result, w)

def update_user_dict():
    test_user_feature = pd.read_csv(PRE + "test", header=None, names=['user_id'])
    all_user_dict = []
    for i in range(PROCESS_NUM + 1):
        print(i)
        with open(mid_pre + "_pred_tmp_result" + str(i) + '.json', 'r', encoding='utf-8') as r:
            data = json.load(r)
            all_user_dict.extend(data)

    # 加载不超过50的部分
    with open(mid_pre + "_less_id_2_vec_result.json", 'r', encoding='utf-8') as r:
        data = json.load(r)
        all_user_dict.extend(data)

    data = pd.DataFrame(all_user_dict)
    # 补齐test中缺失的user
    test_user_set = set(test_user_feature['user_id'])
    user_set = set(data['user_id'])
    part_user_set = test_user_set - user_set
    top_50_hot = pd.read_csv(mid_pre + "top_50_hot.csv")
    count =0
    print("total part:", len(part_user_set))
    for user in part_user_set:
        count += 1
        print(count)
        predict_set = set(top_50_hot.sample(n=50)['item_id'])
        predict_list = map(str, list(predict_set))
        predict_list = ",".join(predict_list)
        all_user_dict.append(dict(user_id=int(user), item_list=predict_list))
    data = pd.DataFrame(all_user_dict)
    # result = pd.merge(self.test_user_feature, data, on='user_id')
    result = test_user_feature.merge(data, how='inner', on='user_id')
    result.to_csv(res_pre + "pred_result", sep="\t", header=None, index=False)

if __name__ == '__main__':

    # result = rating_item(network, user_id='452224162', item_id='250302368')
    #
    # print(result)
    # user_list = [[10000100, 0, 7, 1, 8, [1, 6, 4, 2, 0, 0]],
    #              [10000100, 0, 7, 1, 8, [1, 6, 4, 2, 0, 0]],
    #              [10000100, 0, 7, 1, 8, [1, 6, 4, 2, 0, 0]],
    #              [10000100, 0, 7, 1, 8, [1, 6, 4, 2, 0, 0]]]
    # item_list = [[463911171, 61, 2151, 13395, 2],
    #              [900775433, 61, 2151, 13395, 1.0],
    #              [1048597011, 92, 1982, 104059, 9.0],
    #              [953063189, 107, 452, 80840, 18.0]]

    get_user_item_list(user_dir='../mid_data_random/user_feature_vector_Random.csv',
                       item_dir='../mid_data_random/item_feature_vector_Random.csv',
                       json_dirs=['../mid_data/tmp_result0.json',
                                  '../mid_data/tmp_result1.json',
                                  '../mid_data/tmp_result2.json',
                                  '../mid_data/tmp_result3.json',
                                  '../mid_data/tmp_result4.json'])

    data_df = pd.read_csv('../mid_data_random/predict_list', sep="\|\|\|\|", header=None, names=['user', 'item'])
    print("finish to read file", datetime.datetime.now())
    data_df['user'] = data_df['user'].apply(lambda x: change_user_str_2_vec(x))
    data_df['item'] = data_df['item'].apply(lambda x: change_item_feature_str_2_vec(x))
    print("finish to change structure ", datetime.datetime.now())

    process_list = []
    all_user_num = data_df.shape[0]
    step = all_user_num // PROCESS_NUM
    for i in range(PROCESS_NUM):
        temp_df = data_df.loc[i * step: (i + 1) * step, :]
        p = multiprocessing.Process(target=do_predict,
                                    args=[i,temp_df])
        p.daemon = True
        process_list.append(p)
    for p in process_list:
        p.start()
    for p in process_list:
        p.join()

    temp_df = data_df.loc[PROCESS_NUM * step:, :]
    do_predict(PROCESS_NUM, temp_df)

    update_user_dict()
