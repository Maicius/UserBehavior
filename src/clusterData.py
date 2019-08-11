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
import redis

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
    cluster_column = 'cluster'

    FIND_COMBINE_KEY = 'ub_find_combine'
    freq_item_key = 'freq_item_list'

    def __init__(self, small = True, sup = 0.001):
        # 置信度
        self.support_degree = sup
        self.small = small
        self.user_dict_list = []
        self.process_num = 6
        print("support degree:", self.support_degree)
        if not small:
            self.file_pre = self.res_pre
        else:
            self.file_pre = self.mid_pre
        self.re = self.get_re()

    def get_re(self):
        try:
            pool = redis.ConnectionPool(host='127.0.0.1', port=6379, decode_responses=True)
            return redis.StrictRedis(connection_pool=pool)
        except BaseException as e:
            raise e

    def reserve_file(self):
        pass

    def find_frequent_cate(self, cluster_target='item_id'):
        # 平均每位用户点的小类——42
        # 平均 item_id ———— 157
        # 平均 brand_id 84
        p1 = threading.Thread(target=self.load_train)
        p2 = threading.Thread(target=self.load_item_feature)
        p1.setDaemon(True)
        p2.setDaemon(True)
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        print("waiting for load data...")
        if cluster_target == 'item_id':
            self.train['cluster'] = self.train[cluster_target]
        else:
            print("replace item_id with target property")
            self.item_cate_group = self.item_cate_df.groupby('item_id')
            self.train[self.cluster_column] = self.train['item_id'].apply(lambda x: self.replace_item_with_target(x, cluster_target))
            self.train.drop(self.train[self.train[self.cluster_column] == -1].index, inplace=True)

        # groupby，去除重复的记录
        print("groupby...", datetime.datetime.now())
        self.train['num'] = 1
        self.train = self.train.groupby(by=['user_id', self.cluster_column])['num'].sum().reset_index()
        # self.multi_find_target_id_with_user()
        # 每个target在不同的用户中出现的次数
        self.train['num'] = 1
        self.target_num = self.train.groupby(by=[self.cluster_column])['num'].sum().reset_index()
        total_user_num = len(set(self.train['user_id']))
        print("begin find init frequent item...", datetime.datetime.now())
        support_num = self.support_degree * total_user_num
        print("support num: ", support_num)
        base_frequent_item_list = self.target_num[self.target_num['num'] > support_num][self.cluster_column].values
        print("base freq item num:", len(base_frequent_item_list))
        self.all_frequent_item_list = []

        print("begin find freq item iter...", datetime.datetime.now())
        # 只保留用户列和要聚类的列
        self.train = self.train.loc[:, ['user_id', self.cluster_column]]
        self.cluster_user_group = self.train.groupby(by=self.cluster_column)
        self.cal_frequent_items_iter(base_frequent_item_list, support_num)
        self.all_frequent_item_list = self.re.lrange(self.freq_item_key, 0, -1)
        print("freq item:", self.all_frequent_item_list)

        self.clean_frq_item_set()
        print("after clean:", self.all_frequent_item_list)

        with open(self.file_pre + "freq_item_list.txt", 'w', encoding='utf-8') as w:
            w.writelines(self.all_frequent_item_list)

    def replace_item_with_target(self, item_id, cluster_target):
        try:
            return self.item_cate_group.get_group(item_id)[cluster_target].values[0]
        except KeyError:
            return -1

    def multi_find_target_id_with_user(self):
        print('start cal target_id...', datetime.datetime.now())
        step = self.train.shape[0] // self.process_num
        process_list = []
        for i in range(self.process_num):
            train_df = self.train.iloc[step * i:step * (i + 1), :]
            p = multiprocessing.Process(target=self.cal_brand_id_with_user, args=(train_df, self.item_cate_group, i))
            p.daemon = True
            process_list.append(p)
        for p in process_list:
            p.start()

        for p in process_list:
            p.join()
        print('end cal target_id...', datetime.datetime.now())
        self.update_user_dict()

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
        """
        只保留频繁项集的最大集合，去除子集
        :return:
        """
        for i, item in enumerate(self.all_frequent_item_list):
            item = set(item)

            for j, item2 in enumerate(self.all_frequent_item_list):
                item = set(item)
                if i == j:
                    continue
                else:
                     if item.issubset(item2):
                        try:
                            self.all_frequent_item_list.remove(item)
                        except:
                            print("item have delete", item)

    def cal_frequent_items_iter(self, frequent_item_list, support_num):
        for i in range(2, 6):
            print("group num：" + str(i))
            # 生成备选的频繁子集
            frequent_items = list(combinations(frequent_item_list, i))

            frequent_num = len(frequent_items)
            print("freq_items num:", frequent_num)
            step = frequent_num // self.process_num
            process_list = []
            for i in range(self.process_num):
                frequent_item = frequent_items[i * step: step*(i + 1)]
                p = multiprocessing.Process(target=self.do_find_freq_item, args=[frequent_item, support_num])
                p.daemon = True
                process_list.append(p)
            for p in process_list:
                p.start()
            for p in process_list:
                p.join()
            find_combine = int(self.re.get(self.FIND_COMBINE_KEY))
            if find_combine == 0:
                break


    def do_find_freq_item(self, frequent_items, support_num):
        find_combine = 0
        waste_group = []
        for i, group in enumerate(frequent_items):
            group = set(group)
            print(i, group, datetime.datetime.now())
            is_waste = False
            for wp in waste_group:
                if wp.issubset(group):
                    is_waste = True
                    break
            if not is_waste:
                item_set = set()
                for index, item in enumerate(group):
                    temp_set = set(self.cluster_user_group.get_group(item)['user_id'])
                    if index == 0:
                        item_set = temp_set
                    else:
                        item_set = item_set.intersection(temp_set)
                if len(item_set) > support_num:
                    self.re.lpush(self.freq_item_key, group)
                    find_combine = 1
                else:
                    waste_group.append(group)
        self.re.set('ub_find_combine', find_combine)

    def load_item_feature(self):
        file_name = self.mid_pre + "item_feature.csv" if self.small else self.pre + "item_feature"
        data = pd.read_csv(file_name, sep='\t', header=None,
                           names=['item_id', 'cate_1_id', 'cate_id', 'brand_id', 'price'])
        print("load item feature, shape:", data.shape)
        self.item_cate_df = self.pre_process_item_feature(data)

    @staticmethod
    def pre_process_item_feature(data):
        data.fillna({'cate_1_id': data['cate_1_id'].mode().values[0], 'cate_id': data['cate_id'].mode().values[0],
                     'brand_id': data['brand_id'].mode().values[0], 'price': data['price'].mean()})
        return data

    def load_train(self):
        file_name = self.mid_pre + "train.csv0.01" if self.small else self.mid_pre + "train.csv0.1"
        data = pd.read_csv(file_name, sep='\t', header=None, names=['user_id', 'item_id', 'behavior_type', 'date'])
        print("load train data, shape:", data.shape)
        self.train = data

if __name__ =='__main__':
    t1 = datetime.datetime.now()
    print("begin...", datetime.datetime.now())
    cd = clusterData(small=True)
    cd.find_frequent_cate(cluster_target='cate_id')
    t2 = datetime.datetime.now()
    print("总耗时",(t2 - t1).seconds)