# coding=utf-8
import pandas as pd
import numpy as np

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


    def simple_main(self):
        # self.user_feature = self.load_user_feature()
        self.test_user_feature = pd.read_csv(self.pre + "test", header=None, names=['user_id'])
        self.train = self.load_train()

        # 引入时间权重
        self.train['date'] = self.train['date'].apply(lambda x: self.cal_date_score(x))
        self.train['behavior_type'] = self.train['behavior_type'].map(self.behavior_score)
        self.train['behavior_type'] = self.train['behavior_type'] * self.train['date']
        self.train = self.train.groupby(by=['user_id', 'item_id'])['behavior_type'].sum().reset_index()
        top_50_hot = self.train.groupby(by='item_id')['behavior_type'].mean().reset_index()
        top_50_hot = top_50_hot.sort_values(by=['behavior_type'], ascending=False)
        top_50_hot = top_50_hot.loc[top_50_hot['behavior_type'] > 10]
        pass


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
    sf = Simple_find(small=True)
    sf.simple_main()