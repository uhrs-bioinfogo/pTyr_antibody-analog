from tkinter import  filedialog
import pandas as pd
import os
sequence_path=""#序列的输入路径
AA_frequence_path=""#各个突变位点上氨基酸占比计算后的输出路径

def position_statistic(infile):
    table_list=[]
    position1_dict = {}
    position2_dict = {}
    position3_dict = {}
    position4_dict = {}
    position5_dict = {}
    position6_dict = {}
    position7_dict = {}
    position8_dict = {}
    table_line1 = []
    table_line2 = []
    table_line3 = []
    table_line4 = []
    table_line5 = []
    table_line6 = []
    table_line7 = []
    table_line8 = []
    # table_line9 = []
    # table_line10 = []
    # table_line11 = []
    seq_number=0
    position1_dict = dict(A=0, R=0, N=0, D=0, C=0, Q=0, E=0, G=0, H=0, I=0, L=0, K=0, M=0, F=0, P=0, S=0, T=0, W=0, Y=0,
                          V=0, o=0, a=0, u=0,X=0)
    position2_dict = dict(A=0, R=0, N=0, D=0, C=0, Q=0, E=0, G=0, H=0, I=0, L=0, K=0, M=0, F=0, P=0, S=0, T=0, W=0, Y=0,
                          V=0, o=0, a=0, u=0,X=0)
    position3_dict = dict(A=0, R=0, N=0, D=0, C=0, Q=0, E=0, G=0, H=0, I=0, L=0, K=0, M=0, F=0, P=0, S=0, T=0, W=0, Y=0,
                          V=0, o=0, a=0, u=0,X=0)
    position4_dict = dict(A=0, R=0, N=0, D=0, C=0, Q=0, E=0, G=0, H=0, I=0, L=0, K=0, M=0, F=0, P=0, S=0, T=0, W=0, Y=0,
                          V=0, o=0, a=0, u=0,X=0)
    position5_dict = dict(A=0, R=0, N=0, D=0, C=0, Q=0, E=0, G=0, H=0, I=0, L=0, K=0, M=0, F=0, P=0, S=0, T=0, W=0, Y=0,
                          V=0, o=0, a=0, u=0,X=0)
    position6_dict = dict(A=0, R=0, N=0, D=0, C=0, Q=0, E=0, G=0, H=0, I=0, L=0, K=0, M=0, F=0, P=0, S=0, T=0, W=0, Y=0,
                          V=0, o=0, a=0, u=0,X=0)
    position7_dict = dict(A=0, R=0, N=0, D=0, C=0, Q=0, E=0, G=0, H=0, I=0, L=0, K=0, M=0, F=0, P=0, S=0, T=0, W=0, Y=0,
                          V=0, o=0, a=0, u=0,X=0)
    position8_dict = dict(A=0, R=0, N=0, D=0, C=0, Q=0, E=0, G=0, H=0, I=0, L=0, K=0, M=0, F=0, P=0, S=0, T=0, W=0, Y=0,
                          V=0, o=0, a=0, u=0,X=0)

    for line in infile:

        line=line.strip()
        line_list=list(line)
        position1=line_list[0]
        position2=line_list[1]
        position3=line_list[2]
        position4=line_list[3]
        position5=line_list[4]
        position6=line_list[5]
        position7=line_list[6]
        position8=line_list[7]

        position1_dict[position1] += 1
        position2_dict[position2] += 1
        position3_dict[position3] += 1
        position4_dict[position4] += 1
        position5_dict[position5] += 1
        position6_dict[position6] += 1
        position7_dict[position7] += 1
        position8_dict[position8] += 1

        seq_number += 1
    table_line1 = [str(position1_dict['A'] / seq_number), str(position1_dict['R'] / seq_number),str(position1_dict['N'] / seq_number), str(position1_dict['D'] / seq_number),
                   str(position1_dict['C'] / seq_number), str(position1_dict['Q'] / seq_number),str(position1_dict['E'] / seq_number), str(position1_dict['G'] / seq_number),
                   str(position1_dict['H'] / seq_number), str(position1_dict['I'] / seq_number),str(position1_dict['L'] / seq_number), str(position1_dict['K'] / seq_number),
                   str(position1_dict['M'] / seq_number), str(position1_dict['F'] / seq_number),str(position1_dict['P'] / seq_number), str(position1_dict['S'] / seq_number),
                   str(position1_dict['T'] / seq_number), str(position1_dict['W'] / seq_number),str(position1_dict['Y'] / seq_number), str(position1_dict['V'] / seq_number),
                   str(position1_dict['o'] / seq_number), str(position1_dict['a'] / seq_number),str(position1_dict['u'] / seq_number), str(position1_dict['X'] / seq_number)]
    table_line2 = [str(position2_dict['A'] / seq_number), str(position2_dict['R'] / seq_number),str(position2_dict['N'] / seq_number), str(position2_dict['D'] / seq_number),
                   str(position2_dict['C'] / seq_number), str(position2_dict['Q'] / seq_number),str(position2_dict['E'] / seq_number), str(position2_dict['G'] / seq_number),
                   str(position2_dict['H'] / seq_number), str(position2_dict['I'] / seq_number),str(position2_dict['L'] / seq_number),str(position2_dict['K'] / seq_number),
                   str(position2_dict['M'] / seq_number), str(position2_dict['F'] / seq_number),str(position2_dict['P'] / seq_number),str(position2_dict['S'] / seq_number),
                   str(position2_dict['T'] / seq_number), str(position2_dict['W'] / seq_number),str(position2_dict['Y'] / seq_number),str(position2_dict['V'] / seq_number),
                   str(position2_dict['o'] / seq_number), str(position2_dict['a'] / seq_number),str(position2_dict['u'] / seq_number), str(position2_dict['X'] / seq_number)]
    table_line3 = [str(position3_dict['A'] / seq_number), str(position3_dict['R'] / seq_number),str(position3_dict['N'] / seq_number),str(position3_dict['D'] / seq_number),
                   str(position3_dict['C'] / seq_number), str(position3_dict['Q'] / seq_number),str(position3_dict['E'] / seq_number),str(position3_dict['G'] / seq_number),
                   str(position3_dict['H'] / seq_number), str(position3_dict['I'] / seq_number),str(position3_dict['L'] / seq_number),str(position3_dict['K'] / seq_number),
                   str(position3_dict['M'] / seq_number), str(position3_dict['F'] / seq_number),str(position3_dict['P'] / seq_number),str(position3_dict['S'] / seq_number),
                   str(position3_dict['T'] / seq_number), str(position3_dict['W'] / seq_number),str(position3_dict['Y'] / seq_number),str(position3_dict['V'] / seq_number),
                   str(position3_dict['o'] / seq_number), str(position3_dict['a'] / seq_number),str(position3_dict['u'] / seq_number), str(position3_dict['X'] / seq_number)]
    table_line4 = [str(position4_dict['A'] / seq_number), str(position4_dict['R'] / seq_number),str(position4_dict['N'] / seq_number),str(position4_dict['D'] / seq_number),
                   str(position4_dict['C'] / seq_number), str(position4_dict['Q'] / seq_number),str(position4_dict['E'] / seq_number),str(position4_dict['G'] / seq_number),
                   str(position4_dict['H'] / seq_number), str(position4_dict['I'] / seq_number),str(position4_dict['L'] / seq_number),str(position4_dict['K'] / seq_number),
                   str(position4_dict['M'] / seq_number), str(position4_dict['F'] / seq_number),str(position4_dict['P'] / seq_number),str(position4_dict['S'] / seq_number),
                   str(position4_dict['T'] / seq_number), str(position4_dict['W'] / seq_number),str(position4_dict['Y'] / seq_number),str(position4_dict['V'] / seq_number),
                   str(position4_dict['o'] / seq_number), str(position4_dict['a'] / seq_number),str(position4_dict['u'] / seq_number), str(position4_dict['X'] / seq_number)]
    table_line5 = [str(position5_dict['A'] / seq_number), str(position5_dict['R'] / seq_number), str(position5_dict['N'] / seq_number),
                   str(position5_dict['D'] / seq_number),
                   str(position5_dict['C'] / seq_number), str(position5_dict['Q'] / seq_number),
                   str(position5_dict['E'] / seq_number),
                   str(position5_dict['G'] / seq_number),
                   str(position5_dict['H'] / seq_number), str(position5_dict['I'] / seq_number),
                   str(position5_dict['L'] / seq_number),
                   str(position5_dict['K'] / seq_number),
                   str(position5_dict['M'] / seq_number), str(position5_dict['F'] / seq_number),
                   str(position5_dict['P'] / seq_number),
                   str(position5_dict['S'] / seq_number),
                   str(position5_dict['T'] / seq_number), str(position5_dict['W'] / seq_number),
                   str(position5_dict['Y'] / seq_number),
                   str(position5_dict['V'] / seq_number),
                   str(position5_dict['o'] / seq_number), str(position5_dict['a'] / seq_number),
                   str(position5_dict['u'] / seq_number), str(position5_dict['X'] / seq_number)]
    table_line6 = [str(position6_dict['A'] / seq_number), str(position6_dict['R'] / seq_number),
                   str(position6_dict['N'] / seq_number),
                   str(position6_dict['D'] / seq_number),
                   str(position6_dict['C'] / seq_number), str(position6_dict['Q'] / seq_number),
                   str(position6_dict['E'] / seq_number),
                   str(position6_dict['G'] / seq_number),
                   str(position6_dict['H'] / seq_number), str(position6_dict['I'] / seq_number),
                   str(position6_dict['L'] / seq_number),
                   str(position6_dict['K'] / seq_number),
                   str(position6_dict['M'] / seq_number), str(position6_dict['F'] / seq_number),
                   str(position6_dict['P'] / seq_number),
                   str(position6_dict['S'] / seq_number),
                   str(position6_dict['T'] / seq_number), str(position6_dict['W'] / seq_number),
                   str(position6_dict['Y'] / seq_number),
                   str(position6_dict['V'] / seq_number),
                   str(position6_dict['o'] / seq_number), str(position6_dict['a'] / seq_number),
                   str(position6_dict['u'] / seq_number), str(position6_dict['X'] / seq_number)]
    table_line7 = [str(position7_dict['A'] / seq_number), str(position7_dict['R'] / seq_number),
                   str(position7_dict['N'] / seq_number),
                   str(position7_dict['D'] / seq_number),
                   str(position7_dict['C'] / seq_number), str(position7_dict['Q'] / seq_number),
                   str(position7_dict['E'] / seq_number),
                   str(position7_dict['G'] / seq_number),
                   str(position7_dict['H'] / seq_number), str(position7_dict['I'] / seq_number),
                   str(position7_dict['L'] / seq_number),
                   str(position7_dict['K'] / seq_number),
                   str(position7_dict['M'] / seq_number), str(position7_dict['F'] / seq_number),
                   str(position7_dict['P'] / seq_number),
                   str(position7_dict['S'] / seq_number),
                   str(position7_dict['T'] / seq_number), str(position7_dict['W'] / seq_number),
                   str(position7_dict['Y'] / seq_number),
                   str(position7_dict['V'] / seq_number),
                   str(position7_dict['o'] / seq_number), str(position7_dict['a'] / seq_number),
                   str(position7_dict['u'] / seq_number), str(position7_dict['X'] / seq_number)]
    table_line8 = [str(position8_dict['A'] / seq_number), str(position8_dict['R'] / seq_number),
                   str(position8_dict['N'] / seq_number),
                   str(position8_dict['D'] / seq_number),
                   str(position8_dict['C'] / seq_number), str(position8_dict['Q'] / seq_number),
                   str(position8_dict['E'] / seq_number),
                   str(position8_dict['G'] / seq_number),
                   str(position8_dict['H'] / seq_number), str(position8_dict['I'] / seq_number),
                   str(position8_dict['L'] / seq_number),
                   str(position8_dict['K'] / seq_number),
                   str(position8_dict['M'] / seq_number), str(position8_dict['F'] / seq_number),
                   str(position8_dict['P'] / seq_number),
                   str(position8_dict['S'] / seq_number),
                   str(position8_dict['T'] / seq_number), str(position8_dict['W'] / seq_number),
                   str(position8_dict['Y'] / seq_number),
                   str(position8_dict['V'] / seq_number),
                   str(position8_dict['o'] / seq_number), str(position8_dict['a'] / seq_number),
                   str(position8_dict['u'] / seq_number), str(position8_dict['X'] / seq_number)]

    table_list.append(table_line1)
    table_list.append(table_line2)
    table_list.append(table_line3)
    table_list.append(table_line4)
    table_list.append(table_line5)
    table_list.append(table_line6)
    table_list.append(table_line7)
    table_list.append(table_line8)

    columns=['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                               'Y', 'V', 'TAA', 'TAG', 'TGA','X']
    index=['position1', 'position2', 'position3', 'position4', 'position5', 'position6', 'position7',
                             'position8']
    df=pd.DataFrame(data = table_list, index=index, columns=columns)
    df2 = pd.DataFrame(df.values.T, index=df.columns, columns=df.index)
    return df2

def positiondata_extract(table,number):
    table_data = pd.read_csv(table, sep='\t', header=0)
    position_data=table_data[number]
    return position_data
def tablemerge(number):
    os.chdir('/public/jianghaoqiang/zhaodongping_20210721/fastaq1')
    P1 = pd.read_csv('11.SH2-P1-onlymutation_protein', sep='\t', header=0)
    P2  = pd.read_csv('11.SH2-P2-onlymutation_protein', sep='\t', header=0)
    P3  = pd.read_csv('11.SH2-P3-onlymutation_protein', sep='\t', header=0)
    P4  = pd.read_csv('11.SH2-P4-onlymutation_protein', sep='\t', header=0)
    P5  = pd.read_csv('11.SH2-P5-onlymutation_protein', sep='\t', header=0)
    P6  = pd.read_csv('11.SH2-P6-onlymutation_protein', sep='\t', header=0)
    P7  = pd.read_csv('11.SH2-P7-onlymutation_protein', sep='\t', header=0)
    P8  = pd.read_csv('11.SH2-P8-onlymutation_protein', sep='\t', header=0)
    P9 = pd.read_csv('11.SH2-P9-onlymutation_protein', sep='\t', header=0)
    P10  = pd.read_csv('11.SH2-P10-onlymutation_protein', sep='\t', header=0)
    P11  = pd.read_csv('11.SH2-P11-onlymutation_protein', sep='\t', header=0)
    P12  = pd.read_csv('11.SH2-P12-onlymutation_protein', sep='\t', header=0)
    P13  = pd.read_csv('11.SH2-P13-onlymutation_protein', sep='\t', header=0)

    index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
             'Y', 'V', 'TAA', 'TAG', 'TGA', 'X']
    columns = ['SH2-P1', 'SH2-P2', 'SH2-P3', 'SH2-P4', 'SH2-P5', 'SH2-P6', 'SH2-P7',
               'SH2-P8', 'SH2-P9', 'SH2-P10', 'SH2-P11', 'SH2-P12', 'SH2-P13']
    df3=pd.DataFrame(index=index, columns=columns)
    P1_df = position_statistic(P1)
    P2_df = position_statistic(P2)
    P3_df = position_statistic(P3)
    P4_df = position_statistic(P4)
    P5_df = position_statistic(P5)
    P6_df = position_statistic(P6)
    P7_df = position_statistic(P7)
    P8_df = position_statistic(P8)
    P9_df = position_statistic(P9)
    P10_df = position_statistic(P10)
    P11_df = position_statistic(P11)
    P12_df = position_statistic(P12)
    P13_df = position_statistic(P13)

    df3['SH2-P1']=P1_df[number]
    df3['SH2-P2'] = P2_df[number]
    df3['SH2-P3'] = P3_df[number]
    df3['SH2-P4'] = P4_df[number]
    df3['SH2-P5']= P5_df[number]
    df3['SH2-P6'] = P6_df[number]
    df3['SH2-P7'] = P7_df[number]
    df3['SH2-P8'] = P8_df[number]
    df3['SH2-P9'] = P9_df[number]
    df3['SH2-P10'] = P10_df[number]
    df3['SH2-P11'] = P11_df[number]
    df3['SH2-P12'] = P12_df[number]
    df3['SH2-P13'] = P13_df[number]
    return df3

def main():
    ######################计算各个氨基酸位点位置在各个轮次中的氨基酸频率变化###########################
    # os.chdir('/public/jianghaoqiang/zhaodongping_20210721')
    # POSITION=['position1', 'position2', 'position3', 'position4', 'position5', 'position6', 'position7','position8','position9','position10','position11']
    # for number in POSITION:
    #     DF_POSITION=tablemerge(number)
    #     DF_POSITION.to_csv(number+'position_frequence.csv')
    ##########################################################################################
    IN=pd.read_csv(sequence_path,'\t')
    sequence_list=IN['sequence']
    df=position_statistic(sequence_list)
    df.to_csv(AA_frequence_path)
if __name__ == '__main__':
    main()



