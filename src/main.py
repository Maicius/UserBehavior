import sys
import os

import pandas as pd
import re
import json
import numpy as np

extract_age_pattern = '(?:\[([0-9]{1,2})\,([0-9]{1,2})\])|(?:\>=(60))'
def complete_binary(num, length):
    if len(num) < length:
        num = '0' * (length - len(num)) + num
    return num

class UserBehavior(object):
    pre = "../raw_data/ECommAI_ubp_round1_"
    mid_pre = "../mid_data/"
    real_pre = "../mid_data_random/"
    res_pre = "../result_data/"
    # 对用户的每种行为进行打分
    behavior_score = {
        'clk': 1,
        'collect': 2,
        'cart': 3,
        'buy': 5
    }
    sex_map = {
        'F': '1111',
        'M': '0000'
    }

    def __init__(self, small=True):
        self.small = small

    def main(self, merge=False):
        self.train = self.load_train()

        """
        所有用户数量：987791load_train
        income最大值: 23469
        stage一共10个阶段
        career一共10种，序号1-10，float型
        age 最大值为 >= 60
        """
        self.user_feature = self.load_user_feature()

        """
        所有商品中,数量如下
        item_id = 10786748
        brand_id = 197784
        cate_id = 8381
        cate_1_id = 111
        price的极值不具参考价值
        """
        self.item_feature = self.load_item_feature()
        print(1)
        self.cal_user_item_score()
        print(2)
        self.cal_user_vector()
        print(3)
        self.cal_item_vector()
        # self.user_feature.to_csv(self.mid_pre + 'user_feature_vector.csv')
        # self.item_feature.to_csv(self.mid_pre + 'item_feature_vector.csv')
        if merge:
            print("begin to merge...")
            self.user_item_score = pd.merge(self.user_item_score, self.user_feature,how='inner', on='user_id')
            self.user_item_score = pd.merge(self.user_item_score, self.item_feature, how='inner', on='item_id')
            self.user_item_score.to_csv(self.mid_pre + 'user_item_score_vector_Fix.csv')
        self.user_feature.to_csv(self.mid_pre + "user_feature_fix.csv")
        self.item_feature.to_csv(self.mid_pre + "item_feature_fix.csv")
        return self.user_feature['user_vector'], self.item_feature['item_vector'], self.user_item_score

    # def load_train_vector(self):
    #     item_vector = pd.read_csv(self.mid_pre + 'item_feature0.01.csv')['item_vector']
    #     user_vector = pd.read_csv(self.mid_pre + 'user_feature0.01.csv')['user_vector']
    #     item_vector = list(map(lambda x: list(map(int, x[1:-1].split(','))), item_vector))
    #     user_vector = list(map(lambda x: list(map(int, x[1:-1].split(','))), user_vector))
    #     return user_vector, item_vector
    def load_train_vector(self):
        user_item_score_vector_pd = pd.read_csv(self.mid_pre + 'user_item_score_vector_small0.01.csv')
        user_vector = user_item_score_vector_pd['user_vector']
        item_vector = user_item_score_vector_pd['item_vector']
        score = user_item_score_vector_pd['behavior_type']
        item_vector = list(map(lambda x: list(map(int, x[1:-1].split(','))), item_vector))
        user_vector = list(map(lambda x: list(map(int, x[1:-1].split(','))), user_vector))
        score_vectore = list(map(lambda x: float(x), score))
        return user_vector, item_vector, score_vectore

    def chunk_load_train_vector(self, chunkSize=100000):
        # 添加分段读取逻辑，几十个G的扛不住
        # user_item_score_vector_pd = pd.read_csv(self.mid_pre + 'user_item_score_vector_small0.01.csv', iterator=True)
        user_item_score_vector_pd = pd.read_csv(self.mid_pre + 'user_item_score_vector_Fix.csv', iterator=True)
        loop = True
        chunks = []
        index_i = 0
        while loop:
            index_i += 1
            print("第{}轮读取csv", index_i)
            try:
                chunk = user_item_score_vector_pd.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        print("开始拼接csv")
        df_all = pd.concat(chunks, ignore_index=True)
        print("拼接csv完成")
        print("开始查询列")
        user_vector = df_all['user_vector']
        item_vector = df_all['item_vector']
        score = df_all['behavior_type']
        print("查询完成")
        item_vector = list(map(lambda x: list(map(int, x[1:-1].split(','))), item_vector))
        user_vector = list(map(lambda x: list(map(int, x[1:-1].split(','))), user_vector))
        score_vectore = list(map(lambda x: float(x), score))
        print("训练数据获取完成")

    def chunk_load_train_data(self, chunkSize=100000):
        # 添加分段读取逻辑，几十个G的扛不住
        # 读取列表，而不是向量的结构
        # 返回整个dataframe
        # user_item_score_vector_pd = pd.read_csv(self.real_pre + 'user_item_score_vector_Random2.csv',
        #                                         names=['behavior_type', 'gender',
        #                                                'age', 'career', 'income', 'stage', 'cate_1_id',
        #                                                'cate_id', 'brand_id', 'price'],
        #                                         iterator=True)
        user_item_score_vector_pd = pd.read_csv(self.mid_pre + 'user_item_score_vector_Random2.csv',
                                                names=['behavior_type', 'gender',
                                                       'age', 'career', 'income', 'stage', 'cate_1_id',
                                                       'cate_id', 'brand_id', 'price'],
                                                iterator=True)
        loop = True
        chunks = []
        index_i = 0
        while loop:
            index_i += 1
            print("第{}轮读取csv", index_i)
            try:
                chunk = user_item_score_vector_pd.get_chunk(chunkSize)
                chunks.append(chunk)
            except StopIteration:
                loop = False
                print("Iteration is stopped.")
        print("开始拼接csv")
        df_all = pd.concat(chunks, ignore_index=True)
        print("拼接csv完成")
        target_fields = ['behavior_type']
        features_pd, targets_pd = df_all.drop(target_fields, axis=1), df_all[target_fields]
        print('数据处理完成')
        return features_pd, targets_pd

    def cal_item_vector(self):
        with open(self.mid_pre + "cate_1_dict.json", 'r', encoding='utf-8') as r:
            self.cate_1_dict = json.load(r)
        with open(self.mid_pre + "cate_dict.json", 'r', encoding='utf-8') as r:
            self.cate_dict = json.load(r)
        with open(self.mid_pre + "brand_dict.json", 'r', encoding='utf-8') as r:
            self.brand_dict = json.load(r)

        self.item_feature['cate_1_id'] = self.item_feature['cate_1_id'].apply(lambda x: self.cat_1_to_vector(x))
        self.item_feature['cate_id'] = self.item_feature['cate_id'].apply(lambda x: self.cat_to_vector(x))
        self.item_feature['brand_id'] = self.item_feature['brand_id'].apply(lambda x: self.brand_id_to_vector(x))
        self.item_feature['price'] = self.item_feature['price'].apply(lambda x: self.price_to_vector(x))

        self.item_feature['item_vector'] = self.item_feature['cate_1_id'] + self.item_feature['cate_id'] + self.item_feature[
            'brand_id'] + self.item_feature['price']
        self.item_feature.drop(['cate_1_id', 'cate_id', 'brand_id', 'price'], axis=1, inplace=True)
        self.item_feature['item_vector'] = self.item_feature['item_vector'].apply(lambda x: self.str_vector_2_embedding(x))

    @staticmethod
    def str_vector_2_embedding(str_vector):
        vector = list(str_vector)
        vector = list(map(int, vector))
        return vector

    def cal_user_item_score(self):
        self.train['behavior_type'] = self.train['behavior_type'].map(self.behavior_score)
        date_max = self.train['date'].max()
        date_min = self.train['date'].min()
        # 引入时间权重
        self.train['date'] = self.train['date'].apply(lambda x: self.cal_date_score(x, date_min, date_max))
        self.train['behavior_type'] = self.train['behavior_type'] * self.train['date']
        print("finish time")
        user_item_df = self.train.groupby(by=['user_id', 'item_id'])['behavior_type'].sum().reset_index()
        # user_item_df['hot'] = 1
        # # 计算每个商品的用户数量
        # self.item_user_num = pd.DataFrame(user_item_df.groupby(by=['item_id'])['hot'].sum())
        # # 计算热门惩罚
        # user_item_df['hot_punish'] = user_item_df['item_id'].apply(lambda x:self.cal_hot_punish(x))
        # user_item_df['behavior_type'] = user_item_df['behavior_type'] * user_item_df['hot_punish']
        # print("热门惩罚")
        # user_item_df.drop(['hot', 'hot_punish'], axis=1, inplace=True)
        # 归一化
        score_max = user_item_df['behavior_type'].max()
        score_min = user_item_df['behavior_type'].min()
        score_diff = score_max - score_min
        user_item_df['behavior_type'] = user_item_df['behavior_type'].apply(lambda x: (x - score_min) / score_diff)
        print("归一化")
        self.user_item_score = user_item_df

    @staticmethod
    def cal_date_score(date, date_min, date_max):
        # date_max = 20190620
        date_diff = (date - date_min) / (date_max - date_min)
        f_dt = 1 / (1 + np.e ** (date_diff))
        return f_dt

    def cal_hot_punish(self, item_id):
        user_num = self.item_user_num.loc[item_id]['hot']
        w = 1 / (np.log2(1 + user_num))
        return w

    def cal_user_vector(self):
        # 转化性别为4维向量,不转变为1纬向量是为了平衡权重
        self.user_feature['gender'] = self.user_feature['gender'].map(self.sex_map)
        # 转化年龄为6维向量，因为2的6次方 = 64，年龄的范围是1-60，大于60的按60计算
        self.user_feature['age'] = self.user_feature['age'].apply(lambda x: self.age_to_vector(x))
        # 转化职业为4维向量，因为一共10种职业
        self.user_feature['career'] = self.user_feature['career'].apply(lambda x: self.career_to_vector(x))
        # 将收入转为16维度向量，因为目前收入的最大值一般不会超过过2的16次方,收入应该也是起最大决定作用的参数，所以纬度长一点
        self.user_feature['income'] = self.user_feature['income'].apply(lambda x: self.income_to_vector(x))
        # 将阶段转为10维向量
        self.user_feature['stage'] = self.user_feature['stage'].apply(lambda x: self.stage_to_vector(x))
        # 拼接向量，组成一个40纬的用户向量
        self.user_feature['user_vector'] = self.user_feature['gender'] + self.user_feature['age'] + self.user_feature[
            'career'] + self.user_feature['income'] + self.user_feature['stage']
        # 扔掉转化过的特征，节省内存
        self.user_feature.drop(['gender', 'age', 'career', 'income', 'stage'], inplace=True, axis=1)
        self.user_feature['user_vector'] = self.user_feature['user_vector'].apply(lambda x: self.str_vector_2_embedding(x))

    @staticmethod
    def age_to_vector(age):
        # 将age转化为6位的二进制字符串
        # age 有两种表示方式: [18,20]和>=60
        res = re.findall(extract_age_pattern, age)[0]
        res = list(filter(lambda x: x != '', res))
        # 转化为二进制字符串,再转化为数组，并去掉符号位
        res = bin(int(sum(map(int, res)) / len(res)))[2:]
        # 补齐6位
        return complete_binary(res, 6)

    @staticmethod
    def career_to_vector(career):
        # 将career 转化为4位的二进制
        career_byte = bin(int(career))[2:]
        return complete_binary(career_byte, 4)

    @staticmethod
    def income_to_vector(income):
        income = bin(int(income))[2:]
        return complete_binary(income, 16)

    @staticmethod
    def stage_to_vector(stage):
        stage = stage.split(',')
        stage_list = map(int, stage)
        stage_vector = ['0'] * 10
        for index in stage_list:
            stage_vector[index - 1] = '1'
        return complete_binary("".join(stage_vector), 10)

    def price_to_vector(self, price):
        # 超过4000元的商品在总商品中占比约2.5%
        price = 1023 if price > 1023 else int(price)
        price = bin(price)[2:]
        # 商品价格补齐为10纬
        return complete_binary(price, 10)

    def brand_id_to_vector(self, brand):
        """
        brand的种类一共197784种，约为总商品数的0.02
        brand
        :param brand:
        :return:
        """
        brand = self.brand_dict[str(brand)]
        brand = int((brand - 0) / (197783 - 0) * 8182)
        brand_bin = bin(brand)[2:]
        return complete_binary(brand_bin, 13)

    def cat_1_to_vector(self, cat_1):
        cat_1 = self.cate_1_dict[str(cat_1)]
        cat_1 = bin(cat_1)[2:]
        return complete_binary(cat_1, 7)

    def cat_to_vector(self, cat):
        cat = self.cate_dict[str(cat)]
        cat = bin(cat)[2:]
        return complete_binary(cat, 14)

    def load_train(self):
        file_name = self.mid_pre + "train.csv" if self.small else self.pre + "train"
        data = pd.read_csv(file_name, sep='\t', header=None, names=['user_id', 'item_id', 'behavior_type', 'date'])
        print("load train data, shape:", data.shape)
        return data


    def export_user_item(self):
        """
        导出少量数据到gephi中观察
        :return:
        """
        data = self.train.iloc[:, [0, 1]]
        data.to_csv(self.mid_pre + 'small_user_item.csv', index=False, header=None)

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
        return data

    @staticmethod
    def pre_process_item_feature(data):
        data.fillna({'cate_1_id': data['cate_1_id'].mode().values[0], 'cate_id': data['cate_id'].mode().values[0],
                     'brand_id': data['brand_id'].mode().values[0], 'price': data['price'].mean()})
        return data

if __name__ == '__main__':
    print("begin")
    ub = UserBehavior(small=False)
    ub.main(merge=True)
