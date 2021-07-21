import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import warnings
warnings.filterwarnings('ignore')

#设置图表显示格式
matplotlib.rcParams['font.sans-serif']=['Microsoft YaHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号

#数据提取
data=pd.read_excel(r'C:\Users\ant.zheng\Desktop\dataset\py\ID-Pangle.xlsx',sheet_name='&')
# da=data[data['花费']>0][['创意名称','video_link','花费','cvr','CVR','avgtime']]
df=data[['spend','playnum','avg_time']]

#异常值检测 Q1-1.5(Q3-Q1)  Q3+1.5(Q3-Q1)
# def out(x):
#     q1=x.quantile(0.25)
#     q3=x.quantile(0.75)
#     iqr=q3-q1
#     out_1=q1-1.5*iqr
#     out_3=q3+1.5*iqr
#     return x[x>out_3 & x<out_1]


#剔除重复值
# df.drop_duplicates(subset='需求ID',keep='first',inplace=True)
#广告时长分布
plt.figure(figsize=(10,5))
sns.distplot(df['playnum'])
plt.show()

#K-means聚类
from  sklearn import  preprocessing
import matplotlib
import matplotlib.pyplot  as  plt
from  sklearn.cluster import KMeans

#数据标准化
# min_max_scaler = preprocessing.MinMaxScaler()  #归一化
std_scaler = preprocessing.StandardScaler()
df_norm = std_scaler.fit_transform(df)
#算法评测
# 肘部法则|SSE
distortions =[]
for i in range(1,10):
    km= KMeans(n_clusters=i,
               init='k-means++',
               n_init=10,
               max_iter=300,
               random_state=0)
    km.fit(df_norm)
    distortions.append(km.inertia_)
fig=plt.figure(figsize=(7,4))
plt.plot(range(1,10),distortions,marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortions')

#轮廓系数评测法则
import time
from sklearn.metrics import silhouette_score,calinski_harabasz_score
clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
silhouette_scores = []
#轮廓系数silhouette_scores
print('start time: ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
for k in clusters:
    y_pred = KMeans(n_clusters = k, verbose = 0, n_jobs = -1, random_state=1).fit_predict(df_norm)
    score = silhouette_score(df_norm, y_pred)#silhouette_score：所有样本的轮廓系数的平均值，silhouette_sample：所有样本的轮廓系数
    silhouette_scores.append(score)
print('finish time: ',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
fig1=plt.figure(figsize=(7,4))
plt.plot(clusters, silhouette_scores, '*-')
plt.xlabel('k值')
plt.ylabel('silhouette_score')
plt.show()

#利用肘部法则确定k值，利用轮廓系数评估聚类效果   可选定K=5
# 聚类
plt.scatter(df_norm[:,1], df_norm[:,2], c="red", marker='*')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()
#构造聚类器
clf=estimator = KMeans(n_clusters=5,random_state=1)  # 构造聚类器
clf.fit(df_norm)  # 聚类
#原始数据聚类标签
data['label']=clf.labels_
df['label']=clf.labels_
result=data['label'].value_counts()
print("各个标签的数目\n",result)   #各个标签或类的数目
print("原始数据的分类结果:\n",data.head())

#可视化聚类结果，可分为两种情况（1,按照下述类别 2,通过概率密度图）
label_pred = clf.labels_  # 获取聚类标签
# x0 = df_norm[label_pred == 0]
# x1 = df_norm[label_pred == 1]
# x2 = df_norm[label_pred == 2]
# x3 = df_norm[label_pred == 3]
# x4 = df_norm[label_pred == 4]
# x5 = df_norm[label_pred == 5]
# x6 = df_norm[label_pred == 6]
x0=df[df['label']==0]
x1=df[df['label']==1]
x2=df[df['label']==2]
x3=df[df['label']==3]
x4=df[df['label']==4]
# x5=df[df['label']==5]
# x6=df[df['label']==6]
# x7=df[df['label']==7]


#分离数据，可视化
from mpl_toolkits.mplot3d import Axes3D

x0x=x0['spend']
x0y=x0['playnum']
x0z=x0['avg_time']
x1x=x1['spend']
x1y=x1['playnum']
x1z=x1['avg_time']
x2x=x2['spend']
x2y=x2['playnum']
x2z=x2['avg_time']
x3x=x3['spend']
x3y=x3['playnum']
x3z=x3['avg_time']
x4x=x4['spend']
x4y=x4['playnum']
x4z=x4['avg_time']
# x5x=x5['cvr']
# x5y=x5['CVR']
# x5z=x5['avgtime']
# x6x=x6['cvr']
# x6y=x6['CVR']
# x6z=x6['avgtime']
# x7x=x7['cvr']
# x7y=x7['CVR']
# x7z=x7['avgtime']

# x0x=x0[:, 0]
# x0y=x0[:, 1]
# x0z=x0[:, 2]
# x1x=x1[:, 0]
# x1y=x1[:, 1]
# x1z=x1[:, 2]
# x2x=x2[:, 0]
# x2y=x2[:, 1]
# x2z=x2[:, 2]
# x3x=x3[:, 0]
# x3y=x3[:, 1]
# x3z=x3[:, 2]
# x4x=x4[:, 0]
# x4y=x4[:, 1]
# x4z=x4[:, 2]
# x5x=x5[:, 0]
# x5y=x5[:, 1]
# x5z=x5[:, 2]
# x6x=x6[:, 0]
# x6y=x6[:, 1]
# x6z=x6[:, 2]
data.to_excel(r'C:\Users\ant.zheng\Desktop\两月效果评估_&.xlsx')
# 绘制3D散点图
fig2 = plt.figure()
ax = Axes3D(fig2)
ax.scatter(x0x, x0y, x0z, c='r', label='类0')
ax.scatter(x1x, x1y, x1z, c='g', label='类1')
ax.scatter(x2x, x2y, x2z, c='b', label='类2')
ax.scatter(x3x, x3y, x3z, c='gray', label='类3')
ax.scatter(x4x, x4y, x4z, c='cyan', label='类4')
# ax.scatter(x5x, x5y, x5z, c='yellow', label='类5')
# ax.scatter(x6x, x6y, x6z, c='purple', label='类6')
# ax.scatter(x7x, x7y, x7z, c='#ADFF2F', label='类7')
#绘制图例
ax.legend(loc='best')

#添加坐标轴(顺序是z,y,x)
ax.set_zlabel('spend', fontdict={'size': 12, 'color': 'blue'})
ax.set_ylabel('playnum', fontdict={'size': 12, 'color': 'blue'})
ax.set_xlabel('avgtime', fontdict={'size': 12, 'color': 'blue'})
plt.savefig(r'C:\Users\ant.zheng\Desktop\聚类.png')
plt.show()
print("写入成功")

#绘制聚类后的概率密度图
import matplotlib.pyplot as plt
k=5
n=3
l=0
plt.figure(figsize=(2.5* n, 6.5))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示正负号
for i  in  range(k):
    data=df[df['label']==i]                 #当某类别数据量较少时，输出pdf是否合适
    for j  in  range(3):
        l += 1
        if l>=4:
            l=1
        plt.subplot(n, 1, l)
        try:
            p=data.iloc[:,j].plot(kind='kde', linewidth=2, subplots=True, sharex=False)
            plt.legend()
        except:
            print('结束')
    plt.suptitle(str(i)+'类别客户概率密度图')
    plt.savefig(r'C:\Users\ant.zheng\Desktop\%s.jpg' %(str(i)+'类别'))
    plt.show()