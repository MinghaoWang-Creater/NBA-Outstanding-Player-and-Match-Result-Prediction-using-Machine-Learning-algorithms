# outlier dection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nltk.cluster import kmeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from time import time


# 读取数据函数，将数据读入字典并转换为DataFrame结构
def load_dict_from_file(filepath):
    # 将数据集转化为pd.DataFrame函数适应的特殊格式
    _dict = {}
    count = 0
    key = []
    try:
        with open(filepath, 'r') as dict_file:
            for line in dict_file:
                if count == 0:
                    key = line.strip().split(',')
                    count += 1
                    for i in range(len(key)):
                        _dict[key[i]] = []
                else:
                    value = line.strip().split(',')
                    for i in range(len(key)):
                        _dict[key[i]].append(value[i].strip())

    except IOError as ioerr:
        print("文件 %s 不存在" % (filepath))

    # print(_dict)  # 打印出转化后的特殊格式
    _dict = pd.DataFrame(_dict)  # 将特殊格式转化为DataFrame
    return _dict

# LOF离群点检测
def get_outlier_index(dataset, feature):
    from sklearn.neighbors import LocalOutlierFactor
    model1 = LocalOutlierFactor(n_neighbors=4, contamination=0.1)  # 定义一个LOF模型，异常比例是10%
    model1.fit(feature)
    y1 = model1._predict(feature)
    member = dataset[y1 == -1]
    member.reset_index(inplace=True, drop=True)
    return member

# 孤立森林离群点检测
def get_outlier_index_isolate(dataset, feature):
    from sklearn.ensemble import IsolationForest
    rng = np.random.RandomState(42)
    clf = IsolationForest(max_samples=100 * 2, contamination=0.1, random_state=rng)  # 定义一个孤立森林模型，异常比例为10%
    clf.fit(feature)
    y1 = clf.predict(feature)
    member = dataset[y1 == -1]
    member.reset_index(inplace=True, drop=True)
    return member


# 对原始数据读入处理，获取特征
def data_preprocess(data):
    data = data.fillna(0)
    # 构造有效特征
    data['name'] = data['firstname'].map(str) + str('_') + data['lastname']  # 球员姓名
    data['minutes_ave'] = (data['minutes'].apply(float) / data['gp'].apply(float)).round(2)  # 场均上场时间
    data['pts_ave'] = (data['pts'].apply(float) / data['gp'].apply(float)).round(2)  # 场均得分
    data['turnover'] = pd.to_numeric(data['turnover'], errors='coerce')  # 失误
    data['to_ave'] = (data['turnover'] / data['gp'].apply(float)).round(2)  # 场均失误次数
    data['reb_ave'] = (data['reb'].apply(float) / data['gp'].apply(float)).round(2)  # 场均篮板
    data['asts_ave'] = (data['asts'].apply(float) / data['gp'].apply(float)).round(2)  # 场均助攻
    data['stl_ave'] = (data['stl'].apply(float) / data['gp'].apply(float)).round(2)  # 场均抢断
    data['blk_ave'] = (data['blk'].apply(float) / data['gp'].apply(float)).round(2)  # 场均盖帽
    data['fg_acc'] = (data['fgm'].apply(float) / data['fga'].apply(float)).round(2)  # 投篮命中率
    data['ft_acc'] = (data['ftm'].apply(float) / data['fta'].apply(float)).round(2)  # 罚篮命中率
    data = data.drop_duplicates(['name'], keep='last')  # 删去存在重名的
    data = data.fillna(0)

    # 删去不合理数据
    data = data[data['ft_acc'] <= 1]  # 命中率一定小于等于1
    data = data[data['fg_acc'] <= 1]  # 命中率一定小于等于1
    data = data[data['minutes_ave'] > 3]  # 场均上场时间大于3min才可能是优秀球员

    # 1970年以前未统计抢断、盖帽、失误，应将1970年前后分别处理
    # 根据stl_ave，to_ave，blk_ave三项数据是否全为零划分数据集
    data1 = data[data['stl_ave'].isin([0]) & data['to_ave'].isin([0]) & data['blk_ave'].isin([0])]
    data2 = data[~(data['stl_ave'].isin([0]) & data['to_ave'].isin([0]) & data['blk_ave'].isin([0]))]

    # 构建需要划分的集，st为球员基本信息加特征数据，n_st和n_st1为特征集，分别对应1970前后的选手
    st = ['ilkid', 'name', 'minutes_ave', 'pts_ave', 'to_ave', 'reb_ave', 'asts_ave', 'stl_ave', 'blk_ave',
          'fg_acc', 'ft_acc']
    n_st = ['minutes_ave', 'pts_ave', 'to_ave', 'reb_ave', 'asts_ave', 'stl_ave', 'blk_ave', 'fg_acc', 'ft_acc']
    n_st1 = ['minutes_ave', 'pts_ave', 'reb_ave', 'asts_ave', 'fg_acc', 'ft_acc']

    # 划分data数据
    dataset1 = data1[st]  # 球员基本数据1
    newdataset1 = data1[n_st1]  # 1970前数据
    dataset2 = data2[st]  # 球员基本数据2
    newdataset2 = data2[n_st]  # 1970后数据

    # 重置index
    dataset1.reset_index(inplace=True, drop=True)
    newdataset1.reset_index(inplace=True, drop=True)
    dataset2.reset_index(inplace=True, drop=True)
    newdataset2.reset_index(inplace=True, drop=True)
    return dataset1, newdataset1, dataset2, newdataset2

# 聚类，在离群点中找到杰出球员
def get_outstanding_index_career(newdataset):
    train_x = newdataset
    train_x = train_x.fillna(0)
    df = train_x[['minutes_ave', 'pts_ave', 'fg_acc']].values.tolist()  # 选取用于聚类的特征
    df = np.array(df)
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    # KMeans++实现聚类
    kmeansmodel = KMeans(n_clusters=5, init='k-means++', random_state=0)
    y_kmeans = kmeansmodel.fit_predict(X)

    # 将分类标签加入数据中
    cluster_result = pd.DataFrame({'praise': y_kmeans})
    result = pd.concat([train_x, cluster_result], axis=1)
    choice = np.array([0., 0., 0., 0., 0.])
    for i in range(0, 5):
        choice[i] = np.mean(result[result['praise'] == i]['pts_ave'])
    outstanding_group = np.argmax(choice)  # 根据最高的pts_ave得到优秀球员簇
    res = result[result['praise'] == outstanding_group]
    res.reset_index(inplace=True, drop=True)
    return res

# 主干步骤
# 读入数据
data_player_playoffs_career = load_dict_from_file('player_playoffs_career.txt')   # 调用读取数据函数
data_player_regular_season_career = load_dict_from_file('player_regular_season_career.txt')

# 处理数据，得到特征集
(pr_c1, f_pr_c1, pr_c2, f_pr_c2) = data_preprocess(data_player_regular_season_career)  # 调用数据处理函数，常规赛数据
(po_c1, f_po_c1, po_c2, f_po_c2) = data_preprocess(data_player_playoffs_career)  # 季后赛数据
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

member = {}
t0 = time()

# 利用LOF进行检测：
# member[0] = get_outlier_index(pr_c1, f_pr_c1)
# member[1] = get_outlier_index(pr_c2, f_pr_c2)
# member[2] = get_outlier_index(po_c1, f_po_c1)
# member[3] = get_outlier_index(po_c2, f_po_c2)

# 利用孤立森林进行检测：离群点检测，寻找杰出球员和差劲球员
member[0] = get_outlier_index_isolate(pr_c1,f_pr_c1)  # 调用孤立森林算法函数
member[1] = get_outlier_index_isolate(pr_c2,f_pr_c2)
member[2] = get_outlier_index_isolate(po_c1, f_po_c1)
member[3] = get_outlier_index_isolate(po_c2, f_po_c2)

# 输出时间
t = time() - t0
print("%s :\t%.2fs" % ('共耗时：', t))

# 分别从中找到杰出球员
# 离群点中包含杰出球员和差劲球员，通过聚类，将杰出和差劲分开，选出优秀球员
for i in range(0, 4):
    member[i] = get_outstanding_index_career(member[i])   # 调用离群点中寻找杰出球员的函数

# 将找到的球员进行合并outstandingplayer_1为常规赛杰出球员，outstandingplayer_2为季后赛杰出球员
outstandingplayer_1 = pd.merge(member[0], member[1], how='outer')
outstandingplayer_2 = pd.merge(member[2], member[3], how='outer')
# 将常规赛和季后赛的杰出合并
outstandingplayer = pd.merge(outstandingplayer_1, outstandingplayer_2, how='inner', on=['name', 'ilkid'])

# 将结果转为list
outstandingplayer_regular = outstandingplayer_1['name'].values.tolist()
outstandingplayer_playoff = outstandingplayer_2['name'].values.tolist()
outstandingplayer_all = outstandingplayer['name'].values.tolist()

# 输出最终选拔的优秀球员
print('常规赛中杰出球员共：', len(outstandingplayer_regular), '人：')
print(outstandingplayer_regular)
print('季后赛中杰出球员共：', len(outstandingplayer_playoff), '人：')
print(outstandingplayer_playoff)
print('所有赛季综合中杰出球员共：', len(outstandingplayer_all), '人：')
print(outstandingplayer_all)
