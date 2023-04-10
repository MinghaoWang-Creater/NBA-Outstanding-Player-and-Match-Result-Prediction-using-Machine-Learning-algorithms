import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from time import time

# 读取球队比赛数据
# 读取每场比赛情况（主客场得分），Schedule and Results，并转化为DataFrame
def load_dict_from_file_schedule_results(filepath):
    _dict = {}
    count = 0
    key = []
    try:
        with open(filepath, 'r') as dict_file:
            for line in dict_file:
                if count == 0:
                    key = line.strip().split(',')
                    key[3] = 'PTS1'
                    key[5] = 'PTS2'
                    key[6] = '1'
                    key[7] = '2'
                    count += 1
                    for i in range(len(key)):
                        _dict[key[i]] = []
                    # print(_dict)
                else:
                    value = line.strip().split(',')
                    # print(line)
                    # print(value)
                    for i in range(len(key)):
                        # print(i)
                        _dict[key[i]].append(value[i].strip())
                # (key, value) = line.strip().split(',')
                # _dict[key] = value
    except IOError as ioerr:
        print("文件 %s 不存在" % filepath)
    _dict = pd.DataFrame(_dict)
    return _dict


# 读取球队排名数据Team Ratings，并化为DataFrame
def load_dict_from_file_team_ratings(filepath):
    _dict = {}
    count = 0
    key = []
    try:
        with open(filepath, 'r') as dict_file:
            for line in dict_file:
                if count == 0:
                    key = line.strip().split(',')
                    for i in range(len(key)):
                        _dict[key[i]] = []
                    count += 1
                    # print(_dict)
                else:
                    value = line.strip().split(',')
                    for i in range(len(key)):
                        _dict[key[i]].append(value[i].strip())
                # (key, value) = line.strip().split(',')
                # _dict[key] = value
    except IOError as ioerr:
        print("文件 %s 不存在" % filepath)
    _dict = pd.DataFrame(_dict)
    return _dict


# 计算常规赛天数
def cauculate_day(dataset):
    Day = []
    label = int((dataset['Date'][0].strip().split(' '))[2])
    for i in range(len(dataset['Date'])):
        day = 1
        sign = dataset['Date'][i].strip().split(' ')
        if sign[1] == 'Nov':
            day += 31
        elif sign[1] == 'Dec':
            day += 61
        elif sign[1] == 'Jan':
            day += 92
        elif sign[1] == 'Feb':
            day += 123
        elif sign[1] == 'Mar':
            day += 151
        elif sign[1] == 'Apr':
            day += 182
        elif sign[1] == 'May':
            day += 212
        elif sign[1] == 'Jun':
            day += 243
        # print(int(sign[2]) - label + day)
        Day.append(int(sign[2]) - label + day)
    dataset['Day'] = Day
    return


# 计算累计得分,胜一场积1分，输一场积0分
def points_gain(dataset, name_home):
    score = {}
    h_p = []
    v_p = []
    pts_add = {}
    home = []
    visit = []
    for i in range(len(name_home)):
        pts_add[name_home[i]] = 0
        score[name_home[i]] = 0
    # print(score)
    for i in range(len(dataset.index)):
        # 计算每场净胜分
        pts = dataset['PTS2'][i] - dataset['PTS1'][i]
        pts_add[dataset['Home'][i]] += pts
        pts_add[dataset['Visitor'][i]] -= pts

        # 计算每场积分，获胜积1分，输了积0分
        if dataset['HW'][i] == 1:
            score[dataset['Home'][i]] += 1
        else:
            score[dataset['Visitor'][i]] += 1
        h_p.append(score[dataset['Home'][i]])
        v_p.append(score[dataset['Home'][i]])
        home.append(pts_add[dataset['Home'][i]])
        visit.append(pts_add[dataset['Visitor'][i]])
    dataset['h_pts_add'] = home
    dataset['v_pts_add'] = visit
    dataset['h_score_add'] = h_p
    dataset['v_score_add'] = v_p
    return


# 计算主客场最近三场表现，hm和vm信息，胜为1，负为0
def recent_behavior(dataset, name_home):
    prew = {}
    for i in range(len(name_home)):
        prew[name_home[i]] = []
    hm1 = []
    vm1 = []
    hm2 = []
    vm2 = []
    hm3 = []
    vm3 = []
    for i in range(len(dataset.index)):
        hteam = dataset['Home'][i]
        vteam = dataset['Visitor'][i]
        if len(prew[hteam]) >= 3:
            hm3.append(prew[hteam][-3])
            hm2.append(prew[hteam][-2])
            hm1.append(prew[hteam][-1])
        elif len(prew[hteam]) >= 2:
            hm3.append('N')
            hm2.append(prew[hteam][-2])
            hm1.append(prew[hteam][-1])
        elif len(prew[hteam]) >= 1:
            hm3.append('N')
            hm2.append('N')
            hm1.append(prew[hteam][-1])
        else:
            hm3.append('N')
            hm2.append('N')
            hm1.append('N')

        if len(prew[vteam]) >= 3:
            vm3.append(prew[vteam][-3])
            vm2.append(prew[vteam][-2])
            vm1.append(prew[vteam][-1])
        elif len(prew[vteam]) >= 2:
            vm3.append('N')
            vm2.append(prew[vteam][-2])
            vm1.append(prew[vteam][-1])
        elif len(prew[vteam]) >= 1:
            vm3.append('N')
            vm2.append('N')
            vm1.append(prew[vteam][-1])
        else:
            vm3.append('N')
            vm2.append('N')
            vm1.append('N')

        if dataset['HW'][i] == 1:
            prew[hteam].append(1)
            prew[vteam].append(0)
        else:
            prew[vteam].append(1)
            prew[hteam].append(0)
    dataset['hm1'] = hm1
    dataset['hm2'] = hm2
    dataset['hm3'] = hm3
    dataset['vm1'] = vm1
    dataset['vm2'] = vm2
    dataset['vm3'] = vm3
    return


# 计算比赛周
def cauculate_week(dataset):
    week = []
    for i in range(len(dataset.index)):
        w = dataset['Day'][i] // 7 + 1
        week.append(w)
    dataset['Week'] = week
    return


# 计算周平均值，主客场球队净胜分和积分的平均值
def pts_ave_week(dataset):
    dataset['hPTS_avew'] = (dataset['h_pts_add'].apply(float) / dataset['Week'].apply(float)).round(2)
    dataset['vPTS_avew'] = (dataset['v_pts_add'].apply(float) / dataset['Week'].apply(float)).round(2)
    dataset['hSco_avew'] = (dataset['h_score_add'].apply(float) / dataset['Week'].apply(float)).round(2)
    dataset['vSco_avew'] = (dataset['h_score_add'].apply(float) / dataset['Week'].apply(float)).round(2)
    return


# 删去前几周对球队比赛胜负统计不完全的数据
def delete_somedata(dataset):
    print(~dataset['hm3'].isin(['N']))
    dataset = dataset[~dataset['hm3'].isin(['N'])]
    dataset = dataset[~dataset['vm3'].isin(['N'])]
    return


# 归一化
def convert_1(data):
    data_list = list(data)
    # print(data_list)
    Max = max(data_list)
    Min = min(data_list)
    # print(Max)
    # print(Min)
    return (data - Min) / (Max - Min)


# 得到球队的特征
def get_feature_team(dataset):
    dict = {}
    dataset['MOV/A'] = dataset['MOV/A'].apply(float)
    dataset['ORtg/A'] = dataset['ORtg/A'].apply(float)
    dataset['DRtg/A'] = dataset['DRtg/A'].apply(float)
    dataset['NRtg/A'] = dataset['NRtg/A'].apply(float)
    for i in range(len(dataset.index)):
        dict[dataset['Team'][i]] = []
        dict[dataset['Team'][i]].append(dataset['MOV/A'][i])
        dict[dataset['Team'][i]].append(dataset['ORtg/A'][i])
        dict[dataset['Team'][i]].append(dataset['DRtg/A'][i])
        dict[dataset['Team'][i]].append(dataset['NRtg/A'][i])
    return dict


# 将球队特征加入data中
def insert_features(dataset, dic):
    hMOV = []
    hORtg = []
    hDRtg = []
    hNRtg = []
    vMOV = []
    vORtg = []
    vDRtg = []
    vNRtg = []
    # print(dic)
    for i in range(len(dataset.index)):
        hteam = dataset['Home'][i]
        vteam = dataset['Visitor'][i]
        hMOV.append(dic[hteam][0])
        hORtg.append(dic[hteam][1])
        hDRtg.append(dic[hteam][2])
        hNRtg.append(dic[hteam][3])
        vMOV.append(dic[vteam][0])
        vORtg.append(dic[vteam][1])
        vDRtg.append(dic[vteam][2])
        vNRtg.append(dic[vteam][3])
    dataset['hMOV'] = hMOV
    dataset['hORtg'] = hORtg
    dataset['hDRtg'] = hDRtg
    dataset['hNRtg'] = hNRtg
    dataset['vMOV'] = vMOV
    dataset['vORtg'] = vORtg
    dataset['vDRtg'] = vDRtg
    dataset['vNRtg'] = vNRtg
    return


# 球队比赛数据总处理，获取训练数据
def data_operate(data, data_team):
    data = data.drop(['1', '2'], axis=1)
    data.rename(columns={'Visitor/Neutral': 'Visitor', 'Home/Neutral': 'Home'}, inplace=True)
    name_home = data['Home'].unique()
    # 通过每场比赛分数得到每场比赛结果，存入HW（Home Win？是为1，否为0）
    data['HW'] = ((data['PTS2'].apply(float) - data['PTS1'].apply(float)) > 0).apply(int)
    data['PTS1'] = data['PTS1'].apply(int)
    data['PTS2'] = data['PTS2'].apply(int)
    cauculate_day(data)
    # print(name_home)
    points_gain(data, name_home)
    recent_behavior(data, name_home)
    cauculate_week(data)
    pts_ave_week(data)
    # delete_somedata(data2017)
    data = data[~data['hm3'].isin(['N'])]
    data = data[~data['vm3'].isin(['N'])]
    data.reset_index(inplace=True, drop=True)

    # 获取球队特征
    team_dict = get_feature_team(data_team)
    # print(team_dict)
    # 将球队特征插入data中
    insert_features(data, team_dict)

    # 归一化
    str_convrt = ['hPTS_avew', 'vPTS_avew', 'hMOV', 'hNRtg', 'vMOV', 'vNRtg']
    for i in str_convrt:
        data[i] = convert_1(data[i])

    cols = ['hPTS_avew', 'vPTS_avew', 'hMOV', 'hNRtg', 'vMOV', 'vNRtg', 'hSco_avew', 'vSco_avew']
    for col in cols:
        data[col] = scale(data[col])

    # 构建最终用于训练的特征：
    feature = ['hm1', 'hm2', 'hm3', 'vm1', 'vm2', 'vm3', 'hPTS_avew', 'vPTS_avew', 'hSco_avew', 'vSco_avew', 'hMOV',
               'hNRtg', 'vMOV', 'vNRtg']
    # 特征
    X_feature = data[feature]
    # 标签为HW（主场是否获胜）
    y_label = data['HW']
    # 构建皮尔逊相关热力图
    # plt.rcParams['font.sans-serif']=['SimHei']
    # plt.rcParams['axes.unicode_minus']=False
    # train_data=pd.concat([X_feature,y_label],axis=1)
    # colormap = plt.cm.RdBu
    # plt.figure(figsize=(21,18))
    # plt.title('Pearson Correlation of Features', y=1.05, size=15)
    # sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0,
    #             square=True, cmap=colormap, linecolor='white', annot=True)
    # plt.show()
    # 根据皮尔逊相关热力图删除相关性过高的特征
    X_feature = X_feature.drop(['vSco_avew', 'vMOV', 'hMOV', 'vPTS_avew'], 1)
    train_data = pd.concat([X_feature, y_label], axis=1)
    for i in list(train_data):
        train_data[i] = train_data[i].apply(float)
    return train_data

# 主函数
if __name__ == '__main__':
    dataset = {}
    teamset = {}
    result = {}

    # 得到球队处理数据
    str1 = 'Schedule and Results.txt'
    str2 = 'Team Ratings.txt'
    dataset = load_dict_from_file_schedule_results(str1)
    teamset = load_dict_from_file_team_ratings(str2)
    result = data_operate(dataset, teamset)

    # 划分特征集和标签
    y_label = result['HW']
    X_feature = result.drop(['HW'], axis=1)

    # 划分训练集和测试集
    Train_data, Test_data, Train_y, Test_y = train_test_split(X_feature, y_label, test_size=0.2)

    # 利用SVM进行分类
    from sklearn import svm
    print('SVM运行结果：')
    t0 = time()
    clf = svm.SVC(C=10, kernel='linear', decision_function_shape='ovr')
    clf.fit(Train_data, Train_y, sample_weight=None)
    # 预测比赛结果，并将结果存放在result.csv文件中，主场球队是否获胜（胜为1，负为0）
    prediction = clf.predict(Train_data)
    result = pd.DataFrame({'HW':prediction.astype(np.int32)})
    result.to_csv('E:\\study\\grade2\\machine_learning\\NBA_prediction\\result.csv',index=False)
    # 计算准确性
    acc1 = clf.predict(Train_data) == list(Train_y)
    print('Accuracy_train:%f' % (np.mean(acc1)))
    acc2 = clf.predict(Test_data) == list(Test_y)
    print('Accuracy_test:%f' % (np.mean(acc2)))
    t = time() - t0
    print("%s :\t%.2fs" % ('共耗时：', t))

    # # 逻辑斯蒂回归学习结果
    # print('逻辑回归运行结果：')
    # t0 = time()
    # import sklearn.linear_model as sl
    # logitmodel = sl.LogisticRegression()  # 定义回归模型
    # logitmodel.fit(Train_data, Train_y)  # 训练模型
    # # 预测比赛结果，并将结果存放在result.csv文件中，主场球队是否获胜（胜为1，负为0）
    # prediction = logitmodel.predict(Train_data)
    # result = pd.DataFrame({'HW': prediction.astype(np.int32)})
    # result.to_csv('E:\\study\\grade2\\machine_learning\\NBA_prediction\\result.csv', index=False)
    # # print(classification_report(y_test,logitmodel.predict(x_test)))
    # # print(classification_report(Test_y,logitmodel.predict(Test_data)))
    # # 计算准确性
    # acc1 = logitmodel.predict(Train_data) == list(Train_y)
    # print('Accuracy_train:%f' % (np.mean(acc1)))
    # acc2 = logitmodel.predict(Test_data) == list(Test_y)
    # print('Accuracy_test:%f' % (np.mean(acc2)))
    # t = time() - t0
    # print("%s :\t%.2fs" % ('共耗时：', t))

    # # print(Train_data)
    # # print(Train_y)
    # # print(logitmodel.predict(Train_data))