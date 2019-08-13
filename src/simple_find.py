# coding=utf-8
import pandas as pd
import numpy as np
from random import sample
import json
import datetime
import multiprocessing


class Simple_find(object):
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

    def __init__(self, small=True):
        self.small = small
        self.process_num = 5

    def simple_main(self):
        # self.user_feature = self.load_user_feature()
        self.test_user_feature = pd.read_csv(self.pre + "test", header=None, names=['user_id'])
        self.train = self.load_train()
        # 引入时间权重
        if self.small:
            users = list(set(self.train['user_id'].values))
            user_num = len(users)
        else:
            users = self.test_user_feature['user_id'].values
            user_num = self.test_user_feature.shape[0]

        self.train['date'] = self.train['date'].apply(lambda x: self.cal_date_score(x))
        self.already_buy_df = self.train.loc[self.train['behavior_type'] == 'buy',:]
        self.already_buy_df = self.already_buy_df.groupby(by='user_id')
        self.train['behavior_type'] = self.train['behavior_type'].map(self.behavior_score)
        self.train['behavior_type'] = self.train['behavior_type'] * self.train['date']
        # self.already_buy_df.groupby(by='user_id').get_group(1732029186)
        self.train = self.train.groupby(by=['user_id', 'item_id'])['behavior_type'].sum().reset_index()
        top_50_hot = self.train.groupby(by='item_id')['behavior_type'].mean().reset_index()
        top_50_hot = top_50_hot.sort_values(by=['behavior_type'], ascending=False)
        top_50_hot = top_50_hot.loc[top_50_hot['behavior_type'] > 10,:]
        top_50_hot = top_50_hot.sample(n=1000)
        top_50_hot.to_csv(self.mid_pre + "top_50_hot.csv")
        self.train = self.train.groupby(by=['user_id'])

        step = user_num // self.process_num

        process_list = []
        for i in range(self.process_num):
            p = multiprocessing.Process(target=self.find_item_for_user,
                                        args=(users, top_50_hot, self.already_buy_df, self.train, i, step))
            p.daemon = True
            process_list.append(p)
        for p in process_list:
            p.start()

        for p in process_list:
            p.join()
        self.update_user_dict()

    def find_item_for_user(self, users, top_50_hot,already_buy_df, train, index, step):
        print("进入进程", index)
        result = []
        begin = index * step
        end = (index + 1) * step
        for j in range(begin, end):

            user = users[j]
            print(j, user, datetime.datetime.now())
            try:
                buy_set = set(already_buy_df.get_group(user)['item_id'].values)
            except:
                buy_set = set()
            try:
                click_df = train.get_group(user)
                top_100 = click_df.sort_values(by='behavior_type', ascending=False).head(100)
                click_set = set(top_100['item_id'])
            except:
                click_set = set()
            predict_set = click_set - buy_set
            set_length = len(predict_set)
            if set_length < 50:
                i = 0
                while set_length < 50:
                    n = 50 - set_length
                    predict_set = predict_set | set(top_50_hot.sample(n=n)['item_id'])
                    set_length = len(predict_set)
                    i += 1
                    if i > 10:
                        print("error")
                        break
            # elif set_length > 50:
            #     predict_set = sample(list(predict_set), 50)

            # print(predict_set)
            predict_list = map(str, list(predict_set))
            predict_list = ",".join(predict_list)
            # print(predict_list)
            result.append(dict(user_id=int(user), item_list=predict_list))
        with open(self.mid_pre + "tmp_result" + str(index) + ".json", 'w', encoding='utf-8') as w:
            json.dump(result, w)


    def update_user_dict(self):
        all_user_dict = []
        for i in range(self.process_num):
            with open(self.mid_pre + "tmp_result" + str(i) + '.json', 'r', encoding='utf-8') as r:
                data = json.load(r)
                all_user_dict.extend(data)
        data = pd.DataFrame(all_user_dict)
        # result = pd.merge(self.test_user_feature, data, on='user_id')
        result = self.test_user_feature.merge(data, how='inner', on='user_id')
        result.to_csv(self.res_pre + "result", sep="\t", header=None, index=False)


    @staticmethod
    def cal_date_score(date):
        # date_max = 20190620
        date_min = 20190610
        date_max = 20190620
        date_diff = (date - date_min) / (10)
        f_dt = 1 / (1 + np.e ** (date_diff))
        return f_dt + 1

    def load_user_feature(self):
        file_name = self.mid_pre + "user_feature.csv" if self.small else self.pre + "user_feature"
        data = pd.read_csv(file_name, sep='\t', header=None,
                           names=['user_id', 'gender', 'age', 'edu', 'career', 'income', 'stage'])
        data = self.pre_process_user_feature(data)
        print("load user feature, shape:", data.shape)
        return data

    @staticmethod
    def pre_process_user_feature(data_df):
        # 去掉缺失值太多的列
        data_df.drop("edu", axis=1, inplace=True)
        # 填补缺失值
        data_df.fillna(
            {'age': data_df['age'].mode().values[0], 'career': data_df['career'].mode().values[0],
             'income': data_df['income'].mean(),
             'stage': data_df['stage'].mode().values[0]}, inplace=True)
        return data_df

    def load_item_feature(self):
        file_name = self.mid_pre + "item_feature.csv" if self.small else self.pre + "item_feature"
        data = pd.read_csv(file_name, sep='\t', header=None,
                           names=['item_id', 'cate_1_id', 'cate_id', 'brand_id', 'price'])
        print("load item feature, shape:", data.shape)
        data = self.pre_process_item_feature(data)
        return data.loc[:, ['item_id', 'cate_id', 'brand_id']]

    @staticmethod
    def pre_process_item_feature(data):
        data.fillna({'cate_1_id': data['cate_1_id'].mode().values[0], 'cate_id': data['cate_id'].mode().values[0],
                     'brand_id': data['brand_id'].mode().values[0], 'price': data['price'].mean()})
        return data

    def load_train(self):
        file_name = self.mid_pre + "train.csv" if self.small else self.pre + "train"
        data = pd.read_csv(file_name, sep='\t', header=None, names=['user_id', 'item_id', 'behavior_type', 'date'])
        print("load train data, shape:", data.shape)
        return data


if __name__ =='__main__':
    sf = Simple_find(small=False)
    sf.simple_main()