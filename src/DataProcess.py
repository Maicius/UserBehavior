import json
import pandas as pd
import datetime
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
print(sys.path)
from src.main import UserBehavior

class DataProcess(UserBehavior):
    def __init__(self, small=True):
        UserBehavior.__init__(self, small=small)
        self.pre = "../raw_data/ECommAI_ubp_round1_"
        self.mid_pre_random = "../mid_data_random/"

    def process(self, merge=True):
        print("begin...", datetime.datetime.now())
        self.train = self.load_train()
        print("cal user item score...", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.cal_user_item_score()
        print("cal user vector...", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.cal_user_vector()
        print("cal item vector", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.cal_item_vector()
        if merge:
            print("merge 1...", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            self.user_item_score = pd.merge(self.user_item_score, self.user_feature,how='inner', on='user_id')
            print("merge 2...", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            self.user_item_score = pd.merge(self.user_item_score, self.item_feature, how='inner', on='item_id')
            print("save file...", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            self.user_item_score.to_csv(self.mid_pre_random + 'user_item_score_vector_Random.csv')
            self.user_item_score.drop(["user_id", "item_id"],axis=1, inplace=True)
            self.user_item_score.to_csv(self.mid_pre_random + 'user_item_score_vector_Random2.csv', index=False, header=None)
        self.user_feature.to_csv(self.mid_pre_random + "user_feature_vector_Random.csv")
        self.item_feature.to_csv(self.mid_pre_random + "item_feature_vector_Random.csv")
        print("finish all", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    def calculate_user_map(self, type, save=True):
        age_list = list(set(self.user_feature[type]))
        map = {v: index for index, v in enumerate(age_list)}
        if save:
            with open(self.mid_pre + type + '.json', 'w', encoding='utf-8') as w:
                json.dump(map, w)
        self.user_feature[type] = self.user_feature[type].map(map)
        return map

    @staticmethod
    def pre_pro_stage(stage):
        stage_list = list(map(int, stage.split(',')))
        length = len(stage_list)
        if length < 6:
            stage_list = stage_list + [0] * (6 - length)
        elif length > 6:
            stage_list = stage_list[:6]
        return stage_list

    def cal_user_vector(self):
        self.user_feature = self.load_user_feature()
        sex_map = self.calculate_user_map("gender")
        age_map = self.calculate_user_map("age")
        career_map = self.calculate_user_map("career")
        self.user_feature['income'] = self.user_feature['income'].apply(lambda x: x // 2000)
        income_map = self.calculate_user_map("income")
        self.user_feature['stage'] = self.user_feature['stage'].apply(lambda x: self.pre_pro_stage(x))

    def price_to_embedding(self, price, price_upper):
        price = price_upper if price > price_upper else int(price)
        return price // 100

    def brand_id_to_vector(self, brand):
        """
        brand的种类一共197784种，约为总商品数的0.02
        brand
        :param brand:
        :return:
        """
        brand = self.brand_dict[str(brand)]
        return brand

    def cat_1_to_vector(self, cat_1):
        cat_1 = self.cate_1_dict[str(cat_1)]
        return cat_1

    def cat_to_vector(self, cat):
        cat = self.cate_dict[str(cat)]
        return cat

    def cal_item_vector(self):
        self.item_feature = self.load_item_feature()
        with open(self.mid_pre + "cate_1_dict.json", 'r', encoding='utf-8') as r:
            self.cate_1_dict = json.load(r)
        with open(self.mid_pre + "cate_dict.json", 'r', encoding='utf-8') as r:
            self.cate_dict = json.load(r)
        with open(self.mid_pre + "brand_dict.json", 'r', encoding='utf-8') as r:
            self.brand_dict = json.load(r)
        self.item_feature['cate_1_id'] = self.item_feature['cate_1_id'].apply(lambda x: self.cat_1_to_vector(x))
        self.item_feature['cate_id'] = self.item_feature['cate_id'].apply(lambda x: self.cat_to_vector(x))
        self.item_feature['brand_id'] = self.item_feature['brand_id'].apply(lambda x: self.brand_id_to_vector(x))
        price_upper = self.item_feature['price'].mean() + 1 * self.item_feature['price'].std(axis=0)

        self.item_feature['price'] = self.item_feature['price'].apply(lambda x: self.price_to_embedding(x, price_upper))

if __name__ =='__main__':
    dp = DataProcess(small=False)
    dp.process()