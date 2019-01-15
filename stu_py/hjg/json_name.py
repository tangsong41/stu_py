#功能描述：随机生成一个中文名字

import random
import json
from conf.setting import FIRSTNAME, LASTNAME
def random_name():#定义一个函数
    first_name_list = open(FIRSTNAME, encoding='utf-8')#打开文件，获取文件句柄
    last_name_list = open(LASTNAME, encoding='utf-8')

    first_names = json.load(first_name_list)#从文件中获取用load读取文件，并且把文件中的字符串转换成列表
    last_names = json.load(last_name_list)
    name_all = random.choice(last_names) + random.choice(first_names) + random.choice(first_names)#从列表中获取一个字儿，从另一个列表中获取两个字儿，把字儿都拼接到一起。return name_all
random_name()






last_names = ['赵', '钱', '孙', '李', '周', '吴', '郑', '王', '冯', '陈', '褚', '卫', '蒋', '沈', '韩', '杨', '朱', '秦', '尤', '许',
              '姚', '邵', '堪', '汪', '祁', '毛', '禹', '狄', '米', '贝', '明', '臧', '计', '伏', '成', '戴', '谈', '宋', '茅', '庞',
              '熊', '纪', '舒', '屈', '项', '祝', '董', '梁']

first_names = ['的', '一', '是', '了', '我', '不', '人', '在', '他', '有', '这', '个', '上', '们', '来', '到', '时', '大', '地', '为',
               '子', '中', '你', '说', '生', '国', '年', '着', '就', '那', '和', '要', '她', '出', '也', '得', '里', '后', '自', '以',
               '乾', '坤', '']

res = json.dumps(first_names, ensure_ascii=False)#把frist_names列表转换成json字符串
print(res)#打印出来是一个字符串类型的列表
print(type(res))#打印res类型是字符串
f = open('first_names.json', 'w', encoding='utf8')#打开文件，赋给f文件句柄
f = open('last_names.json', 'w', encoding='utf8')#打开文件，赋给f文件句柄
json.dump(first_names, f, ensure_ascii=False, indent=10)#直接写入文件了，不需要再f.write，写入以后还有缩进，是json格式
json.dump(first_names, f, ensure_ascii=False, indent=10)#直接写入文件了，不需要再f.write，写入以后还有缩进，是json格式