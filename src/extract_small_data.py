import pandas as pd
import json



def get_small_data(type, num=1000):
    pre = "../raw_data/ECommAI_ubp_round1_"
    small_file = '../mid_data/' + type + ".csv"
    small_data = []

    # 如果num为-1，则取全部
    if num == -1:
        num = 1000000000000
    with open(pre + type, 'r', encoding='utf-8') as r:
        for index, line in enumerate(r):
            if index > num:
                break
            else:
                print(line)
                small_data.append(line)
    with open(small_file, 'w', encoding='utf-8') as w:
        w.writelines(small_data)

def extract_item_cat_1_dict():
    file_name = "../raw_data/ECommAI_ubp_round1_" + "item_feature"
    data = pd.read_csv(file_name, sep='\t', header=None,
                       names=['item_id', 'cate_1_id', 'cate_id', 'brand_id', 'price'])
    cat_1 = list(set(data['cate_1_id']))
    cat_1_dict = {}
    for index, item in enumerate(cat_1):
        cat_1_dict[item] = index
    with open("../mid_data/cate_1_dict.json", 'w', encoding='utf-8') as w:
        json.dump(cat_1_dict, w)

    cat = list(set(data['cate_id']))
    cat_dict = {}
    for index, item in enumerate(cat):
        cat_dict[item] = index
    with open("../mid_data/cate_dict.json", 'w', encoding='utf-8') as w:
        json.dump(cat_dict, w)

    brand = list(set(data['brand_id']))
    brand_dict = {}
    for index, item in enumerate(brand):
        brand_dict[item] = index
    with open("../mid_data/brand_dict.json", 'w', encoding='utf-8') as w:
        json.dump(brand_dict, w)


if __name__ =='__main__':
    # extract_item_cat_1_dict()
    # 提取小数据集，num=-1表示取全部
    get_small_data("train", num=10000)
    get_small_data("item_feature", num=10000)
    get_small_data("user_feature", num=10000)