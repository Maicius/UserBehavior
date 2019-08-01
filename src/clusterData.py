# coding=utf-8
import pandas as pd
from itertools import combinations
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
import json
import datetime
import multiprocessing
import threading

class clusterData(object):
    pre = "../raw_data/ECommAI_ubp_round1_"
    mid_pre = "../mid_data/"
    real_pre = "../mid_data_random/"
    res_pre = "../result_data/"
    behavior_score = {
        'clk': 1,
        'collect': 2,
        'cart': 3,
        'buy': 5
    }

    def __init__(self, small = True, sup = 0.1):
        # 置信度
        self.support_degree = sup
        self.small = small
        self.user_dict_list = []
        self.process_num = 8
        if not small:
            self.file_pre = self.res_pre
        else:
            self.file_pre = self.mid_pre

    def reserve_file(self):
        pass

    def find_frequent_cate(self):
        # 平均每位用户点的小类——42
        # 平均 item_id ———— 157
        # 平均 brand_id 84

        p1 = threading.Thread(target=self.load_item_feature)
        p2 = threading.Thread(target=self.load_train)
        p1.setDaemon(True)
        p2.setDaemon(True)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        item_cate_list = self.item_cate_df.values
        # self.item_cate_dict = {item[0]: item[1] for item in item_cate_list}
        # self.cate_item_dict = {item[1]: item[0] for item in item_cate_list}
        self.item_brand_dict = {item[0]: item[1] for item in item_cate_list}
        # self.train['cate_id'] = self.train['item_id'].apply(lambda x: self.replace_item_id(x))
        print("replace...", datetime.datetime.now())
        self.train['cate_id'] = self.train['item_id'].apply(lambda x: self.replace_item_id_with_brand(x))
        self.train['num'] = 1
        self.item_cate_group = self.item_cate_df.groupby(by='brand_id')
        self.train.drop(self.train[self.train['cate_id'] == -1].index, inplace=True)
        # 每个用户对应的品牌，去重
        print("groupby...", datetime.datetime.now())
        self.train = self.train.groupby(by=['user_id', 'cate_id'])['num'].sum().reset_index()

        process_list = []
        step = self.train.shape[0] // self.process_num
        self.total_user_num = len(set(self.train['user_id']))
        print('start cal brand...', datetime.datetime.now())
        for i in range(self.process_num):
            train_df = self.train.iloc[step * i:step * (i + 1), :]
            p = multiprocessing.Process(target=self.cal_brand_id_with_user, args=(train_df, self.item_cate_group, i))
            p.daemon = True
            process_list.append(p)
        for p in process_list:
            p.start()

        for p in process_list:
            p.join()
        print('end cal brand...', datetime.datetime.now())
        self.update_user_dict()
        print("find brand_id with user...", datetime.datetime.now())
        # user_dict = self.cal_brand_id_with_user(df=self.train, user_set=self.user_set, item_cate_df=self.item_cate_df)
        # 每个品牌在不同的用户中出现的次数
        self.item_num = self.train.groupby(by=['cate_id'])['num'].sum().reset_index()
        support_num = self.support_degree * self.total_user_num
        base_frequent_item_list = self.item_num[self.item_num['num'] > support_num]['cate_id'].values
        self.all_frequent_item_list = []
        print(base_frequent_item_list)
        self.cal_frequent_items_iter(base_frequent_item_list, support_num)
        print("freq item:", self.all_frequent_item_list)
        self.clean_frq_item_set()
        print("after clean:", self.all_frequent_item_list)

        with open(self.file_pre + "freq_item_list.txt", 'w', encoding='utf-8') as w:
            w.writelines(self.all_frequent_item_list)


    def update_user_dict(self):
        all_user_dict = {}
        for i in range(self.process_num):
            with open(self.file_pre + "user_brand_item_dict" + str(i) + '.json', 'r') as r:
                data = json.load(r)
                all_user_dict.update(data)
        with open(self.file_pre + "all_user_brand_item_dict.json", 'w', encoding='utf-8') as w:
            json.dump(all_user_dict, w)

    def cal_brand_id_with_user(self, df, item_cate_group, i):
        print('process' + str(i) + 'start...', datetime.datetime.now())
        user_dict = {}
        count = 0
        # 按用户聚类，cate_id为每个用户的品牌列表
        self.train_group = df.groupby(by=['user_id'])
        user_set = self.train_group.user_id.indices.keys()
        df = df.groupby(by=['user_id'])

        for user in user_set:
            count += 1
            if count % 1000 == 0:
                print(count, user, datetime.datetime.now())
                with open(self.mid_pre + "user_brand_item_dict" + str(i) + '.json', 'w', encoding='utf-8') as w:
                    json.dump(user_dict, w)
            cate_list = df.get_group(user)['cate_id']
            item_list = []
            for cate in cate_list:
                item_list += list(item_cate_group.get_group(cate)['item_id'])
            # temp_df = item_cate_group.loc[item_cate_group.brand_id.isin(cate_list)]['item_id']
            # item_list = temp_df.values
            # item_index = temp_df.index
            item_list = list(map(int, item_list))
            user_dict[user] = item_list
            # item_cate_group = item_cate_group.drop(index=item_index, axis=0)

        with open(self.file_pre + "user_brand_item_dict" + str(i) + '.json', 'w', encoding='utf-8') as w:
            json.dump(user_dict, w)
        return user_dict

    def clean_frq_item_set(self):
        for i, item in enumerate(self.all_frequent_item_list):
            for j, item2 in enumerate(self.all_frequent_item_list):
                if i == j:
                    continue
                else:
                     if item.issubset(item2):
                        try:
                            self.all_frequent_item_list.remove(item)
                        except:
                            print("item have delete", item)

    def cal_frequent_items_iter(self, frequent_item_list, support_num):
        waste_group = []
        for i in range(2, 6):
            print("group num：" + str(i))
            frequent_items = combinations(frequent_item_list, i)
            find_combine = False
            for group in frequent_items:
                group = set(group)
                is_waste = False
                for wp in waste_group:
                    if wp.issubset(group):
                        is_waste = True
                        break
                if not is_waste:
                    item_set = set()
                    for index, item in enumerate(group):
                        temp_set = set(self.train.loc[self.train['cate_id'] == item, 'user_id'])
                        if index == 0:
                            item_set = temp_set
                        else:
                            item_set = item_set.intersection(temp_set)
                    if len(item_set) > support_num:
                        freq_list = list(group)
                        self.all_frequent_item_list.append(freq_list)
                        find_combine = True
                    else:
                        waste_group.append(group)
            if not find_combine:
                break


    def replace_item_id(self, x):
        try:
            return self.item_cate_dict[x]
        except:
            return -1

    def replace_item_id_with_brand(self, x):
        try:
            return self.item_brand_dict[x]
        except:
            return -1

    def load_item_feature(self):
        file_name = self.mid_pre + "item_feature.csv" if self.small else self.pre + "item_feature"
        data = pd.read_csv(file_name, sep='\t', header=None,
                           names=['item_id', 'cate_1_id', 'cate_id', 'brand_id', 'price'])
        print("load item feature, shape:", data.shape)
        data = self.pre_process_item_feature(data)
        self.item_cate_df = data.loc[:, ['item_id', 'brand_id']]

    @staticmethod
    def pre_process_item_feature(data):
        data.fillna({'cate_1_id': data['cate_1_id'].mode().values[0], 'cate_id': data['cate_id'].mode().values[0],
                     'brand_id': data['brand_id'].mode().values[0], 'price': data['price'].mean()})
        return data

    def load_train(self):
        file_name = self.mid_pre + "train.csv0.01" if self.small else self.pre + "train"
        data = pd.read_csv(file_name, sep='\t', header=None, names=['user_id', 'item_id', 'behavior_type', 'date'])
        print("load train data, shape:", data.shape)
        self.train = data

if __name__ =='__main__':
    t1 = datetime.datetime.now()
    print("begin...", datetime.datetime.now())
    cd = clusterData(small=False)
    cd.find_frequent_cate()
    t2 = datetime.datetime.now()
    print("总耗时",(t2 - t1).seconds)