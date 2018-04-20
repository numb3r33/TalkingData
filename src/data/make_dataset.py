import pandas as pd
import numpy as np
import gc

from collections import Counter


phone_brand_map = {
    '三星': 'samsung',
    '天语': 'Ktouch',
    '海信': 'hisense',
    '联想': 'lenovo',
    '欧比': 'obi',
    '爱派尔': 'ipair',
    '努比亚': 'nubia',
    '优米': 'youmi',
    '朵唯': 'dowe',
    '黑米': 'heymi',
    '锤子': 'hammer',
    '酷比魔方': 'koobee',
    '美图': 'meitu',
    '尼比鲁': 'nibilu',
    '一加': 'oneplus',
    '优购': 'yougo',
    '诺基亚': 'nokia',
    '糖葫芦': 'candy',
    '中国移动': 'ccmc',
    '语信': 'yuxin',
    '基伍': 'kiwu',
    '青橙': 'greeno',
    '华硕': 'asus',
    '夏新': 'panosonic',
    '维图': 'weitu',
    '艾优尼': 'aiyouni',
    '摩托罗拉': 'moto',
    '乡米': 'xiangmi',
    '米奇': 'micky',
    '大可乐': 'bigcola',
    '沃普丰': 'wpf',
    '神舟': 'hasse',
    '摩乐': 'mole',
    '飞秒': 'fs',
    '米歌': 'mige',
    '富可视': 'fks',
    '德赛': 'desci',
    '梦米': 'mengmi',
    '乐视': 'lshi',
    '小杨树': 'smallt',
    '纽曼': 'newman',
    '邦华': 'banghua',
    'E派': 'epai',
    '易派': 'epai',
    '普耐尔': 'pner',
    '欧新': 'ouxin',
    '西米': 'ximi',
    '海尔': 'haier',
    '波导': 'bodao',
    '糯米': 'nuomi',
    '唯米': 'weimi',
    '酷珀': 'kupo',
    '谷歌': 'google',
    '昂达': 'ada',
    '聆韵': 'lingyun',
    '小米': 'millet',
    '华为': 'huawei',
    '魅族': 'meizu',
    '酷派': 'cool_pie',
    '中兴': 'zte',
    '金立': 'gionee',
    '索尼': 'sony',
    '酷比': 'cool_ratio',
    '康佳': 'konka',
    '奇酷': 'queer',
    '欧博信': 'opson',
    '亿通': 'yitong',
    '金星数码': 'venus_digital',
    '广信': 'guang_xin',
    '至尊宝': 'supreme_treasure',
    '百立丰': 'parkson'
}


def convert_phone_brand(pb):
    return phone_brand_map[pb] if pb in phone_brand_map else pb


def load_data():
    train  = pd.read_csv('../../data/raw/gender_age_train.csv', dtype={'device_id': str})
    test   = pd.read_csv('../../data/raw/gender_age_test.csv', dtype={'device_id': str})
    events = pd.read_csv('../../data/raw/events.csv', 
                         dtype={'event_id': str, 'device_id': str},
                         parse_dates=['timestamp']
                        )
    apps   = pd.read_csv('../../data/raw/app_events.csv', dtype={'app_id': str, 'event_id': str})
    device = pd.read_csv('../../data/raw/phone_brand_device_model.csv', dtype={'device_id': str},
                         converters={'phone_brand': convert_phone_brand},
                         encoding='utf-8'
                        )
    sub    = pd.read_csv('../../data/raw/sample_submission.csv')

    return train, test, events, apps, device, sub


def merge_with_device_data(events_train, events_test, device):
    events_train = events_train.merge(device, left_on='device_id',
                                      right_on='device_id',
                                      how='inner'
                                     )

    events_test  = events_test.merge(device, left_on='device_id',
                                     right_on='device_id',
                                     how='inner'
                                    )

    return events_train, events_test

def merge_with_events_data(train, test, events):
    events_train = events.merge(train, 
                            left_on=['device_id'], 
                            right_on=['device_id'],
                            how='inner'
                           )

    events_test  = events.merge(test,
                                left_on=['device_id'],
                                right_on=['device_id'],
                                how='inner'
                               )

    return events_train, events_test