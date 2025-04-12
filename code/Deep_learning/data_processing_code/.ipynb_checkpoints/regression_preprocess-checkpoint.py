import pandas as pd
import numpy as np
import math
R2 = open("", "r")
R4 = open("","r")
merge = open("","r")
out_regression = open("","w")


# 拷贝数总和，此处给定一个初始值
R4_number = 0
R2_number = 0
r2_seq_number = 0
r4_seq_number=0
merge_seq_number=0
R2_dict={}
R4_dict={}
merge_dict={}
out_regression.write('sequence' + '\t' + 'value' + '\n')
#以两轮之间的富集值作为label，例如：某序列的富集值以该序列在R4中的比列与该序列在R2中的比值的比值，来取log10，以此log值来做为该序列的label值
for R2_line in R2:
    R2_line = R2_line.strip()
    R2_split = R2_line.split('\t\t')
    r2_seq_number = float(R2_split[1])+10
    R2_dict[R2_split[0]] = r2_seq_number
    R2_number+=float(R2_split[1])
for R4_line in R4:
    R4_line = R4_line.strip()
    R4_split = R4_line.split('\t\t')
    r4_seq_number = float(R4_split[1])+10
    R4_dict[R3_split[0]] = r4_seq_number
    R4_number+=float(R4_split[1])
    # R3_sort += 1
for merge_line in merge:
    merge_line = merge_line.strip()
    merge_split = merge_line.split('\t\t')
    merge_seq_number = 10
    merge_dict[merge_split[0]] = merge_seq_number
for merge_key in merge_dict:
    if merge_key not in R4_dict:
        R4_dict[merge_key] = 10
        # R3_number += 100
    if merge_key not in R2_dict:
        R2_dict[merge_key] = 10
        # R2_number += 100

for key in merge_dict:
    R4_frequence = R4_dict[key] / R4_number
    R2_frequence = R2_dict[key] / R2_number
    R4_R2_ratio = R4_frequence / R2_frequence
    label = np.log10(R4_R2_ratio)
    label_format = format(label, 'f')
    out_regression.write(key + '\t' + str(label_format) + '\n')

