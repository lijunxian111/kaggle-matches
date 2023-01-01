# -*- coding: utf-8 -*-
import re


# 这个文件是专门处理str类型的文件，
# 主要目的是去掉一段话里的非英文的内容，去掉url，去掉特殊字符如\n,\t,\r,x00这样的特殊字符
# 还有就是去掉文字中的所有符号，
# 把文字变成小写。

def get_english(dd):
        st = ""
        for k in dd.split():
            if len(re.findall("[^a-zA-Z\d.]", k)) == 0:
                st = st + " " + k
        return st

def process_data(data) -> str:
        # 去掉url
        data_first = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%|-)*\b', '', data, flags=re.MULTILINE)
        # 去掉所有的符号，把大写改为小写。
        data_second = data_first.replace(r"\n", " ").replace("?", ' ') \
            .replace("/", ' ').replace(",", ' ').replace("\\", ' '). \
            replace("~", ' ').replace("+", ' ').replace("=", ' ') \
            .replace("!", ' ').lower().replace("#", ' ').replace("@", ' ').replace(r"""""", '') \
            .replace("$", ' ').replace("%", ' ').replace("(", ' ').replace(r"\r", ' ') \
            .replace(")", ' ').replace("-", ' ').replace("_", '').replace(":", ' ') \
            .replace(";", ' ').replace("'", ' ').replace("{", ' ').replace("}", ' ') \
            .replace("[", ' ').replace("]", ' ').replace("|", ' ').replace("*", ' ') \
            .replace(">", ' ').replace("<", ' ').replace("$", " ").replace("^", ' ') \
            .replace(r"\t", ' ')
        # 去掉x0z这类的东西
        data_three = re.sub(r'x[0-9][a-zA-Z.\d]*', '', data_second, flags=re.MULTILINE)
        # 去掉非英文和数字的部分
        data_four = get_english(data_three).replace(".", " ")
        return data_four
