def get_small_data(type, num=1000):
    pre = "../raw_data/ECommAI_ubp_round1_"
    small_file = '../mid_data/' + type + ".csv"
    small_data = []
    with open(pre + type, 'r', encoding='utf-8') as r:
        for index, line in enumerate(r):
            if index > num:
                break
            else:
                print(line)
                small_data.append(line)
    with open(small_file, 'w', encoding='utf-8') as w:
        w.writelines(small_data)

if __name__ =='__main__':
    get_small_data("train", num=10000)
    get_small_data("item_feature", num=10000)
    get_small_data("user_feature", num=10000)