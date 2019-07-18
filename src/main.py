import pandas as pd
import re
from src.util import extract_age_pattern, complete_binary


class UserBehavior(object):
    pre = "../raw_data/ECommAI_ubp_round1_"
    mid_pre = "../mid_data/"
    res_pre = "../result_data/"
    behavior_score = {
        'clk': 1,
        'collect': 3,
        'cart': 5,
        'buy': 10
    }
    sex_map = {
        'F': '1111',
        'M': '0000'
    }

    def __init__(self, small=True):
        self.small = small
        self.train = self.load_train()
        self.user_feature = self.load_user_feature()
        self.item_feature = self.load_item_feature()
        # self.export_user_item()
        pass

    def main(self):
        self.cal_user_item_score()
        self.cal_user_vector()

        pass

    def cal_user_item_score(self):
        self.train['behavior_type'] = self.train['behavior_type'].map(self.behavior_score)
        user_item_df = self.train.groupby(by=['user_id', 'item_id'])['behavior_type'].sum().reset_index()
        # user_item_df.sort_values(by='behavior_type', ascending=False, inplace=True)
        user_item_df.to_csv(self.res_pre + 'user_item_score.csv')

    def cal_user_vector(self):
        # 转化性别为4维向量,不转变为1纬向量是为了平衡权重
        self.user_feature['gender'] = self.user_feature['gender'].map(self.sex_map)
        # 转化年龄为6维向量，因为2的6次方 = 64
        self.user_feature['age'] = self.user_feature['age'].apply(lambda x: self.age_to_vector(x))
        # 转化职业为4维向量，因为一共10种职业
        self.user_feature['career'] = self.user_feature['career'].apply(lambda x: self.career_to_vector(x))
        # 将收入转为16维度向量，因为目前收入的最大值一般不会超过过2的16次方,收入应该也是起最大决定作用的参数，所以纬度长一点
        self.user_feature['income'] = self.user_feature['income'].apply(lambda x: self.income_to_vector(x))
        # 将阶段转为10维向量
        self.user_feature['stage'] = self.user_feature['stage'].apply(lambda x: self.stage_to_vector(x))

        # 拼接向量，组成一个40纬的用户向量
        self.user_feature['vector'] = self.user_feature['gender'] + self.user_feature['age'] + self.user_feature[
            'career'] + self.user_feature['income'] + self.user_feature['stage']

        # 扔掉转化过的特征，节省内存
        self.user_feature.drop(['gender', 'age', 'career', 'income', 'stage'], inplace=True, axis=1)

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
        # 将career 转化为 4位的二进制
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

    def pre_process_user_feature(self, data_df):
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
        return data


if __name__ == '__main__':
    ub = UserBehavior()
    ub.main()
