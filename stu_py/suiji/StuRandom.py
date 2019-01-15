#  coding=utf-8


import random
import random as r

def  randomName():
   first_name = ["王", "李", "张", "刘", "赵", "蒋", "孟", "陈", "徐", "杨", "沈", "马", "高", "殷", "上官", "钟", "常"]
   second_name = ["伟", "华", "建国", "洋", "刚", "万里", "爱民", "牧", "陆", "路", "昕", "鑫", "兵", "硕", "志宏", "峰", "磊", "雷", "文","明浩", "光", "超", "军", "达"]
   name = r.choice(first_name) + r.choice(second_name)
   print(name)
   return name


#def randomPhone():



xingming=randomName();
print(xingming)


