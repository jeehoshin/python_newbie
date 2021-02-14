#!/usr/bin/env python
# coding: utf-8


# 스파이더에서 셀을 나누려면 #%%을 적어주면 된다.

#%% : 샘플 데이터 생성

import numpy as np
import pandas as pd
np.random.seed(1234)

#생성된 난수에 대해 정규화를 진행한다.
data1 = pd.DataFrame(
    data = {'grade' : np.random.rand(10000)*100})
data1.index.name = 'index'
data1['group'] = 0
data1.to_csv('C:/Users/jiho0/OneDrive/바탕 화면/SAS/SAS_CODE/참고자료/sample_data/sample.txt', sep = ' ', encoding = 'cp949')


#%% : 평균 50인 정규분포에 대해 0.05의 누적확률분포가 0.0005일 표준편차를 찾음.
from scipy.stats import norm

for i in np.arange(10,16,0.00001) : 
    if norm(50,i).cdf(x=0) >= 0.0005 : #0점에 할당되는 누적확률이 0.05%이도록!
        print(i)
        break

# In[3]:
from scipy.stats import norm

pctl = []
i = 15.179949999803899
for k in range(0,102,2) : 
    pctl.append(round(norm(50,i).cdf(x = k), ndigits = 4))

print(pctl)

# In[4]:

cut1 = data1.quantile(q = pctl)
cut1['group'] = np.arange(1,52,1)

# cut1 = cut1.sort_values(by = ['grade'], axis = 0, ascending = False)
print(cut1.head(5))


# In[15]: lag데이터 생성 
cut2 = cut1.shift(-1)
cut2['group'] = cut2['group'].fillna(52)
cut2['grade'] = cut2['grade'].fillna(100) 
cut2.columns = ['upper', 'group']

cut3 = cut1.join(cut2['upper'])
del cut1, cut2
# In[ ] : 등급 컷에 맞는 그룹을 할당

for a in data1.index : 
    val = data1.loc[a, 'grade']
    for b in cut3.index : 
        cut_upper = cut3.loc[b, 'upper']
        cut_lower = cut3.loc[b, 'grade']
        group = cut3.loc[b, 'group']
        if val > cut_lower and val <= cut_upper : 
            data1.loc[a, 'group'] = group
            
#%% : group by 하여 그룹별 최소값 최대값 추출 
data2 = data1.groupby('group').max()
data2.columns = ['max'] 

#%% : lag 칼럼 생성
max_lag = data2['max'].shift(1)
max_lag.fillna(0)
data3 = pd.concat([data2, max_lag], axis = 1)
data3.columns = ['max', 'max_lag']
del max_lag
#%% Base 칼럼 생성
base1 = np.arange(0,100,2)
base1 = base1.tolist()
base1.insert(0, np.nan)
base1.append(np.nan)
base2 = pd.Series(base1)

data4 = pd.concat([data3, base2], axis = 1)
data4.sort_index(inplace = True)
data4.columns = ['max', 'max_lag', 'base']
del base1, base2

#%% gradient 생성 
data4['gradient'] = 2 / (data4['max'] - data4['max_lag'])

#%% 각 그룹별 선형보간 계산 진행
for a in data1.index : 
    val  = data1.loc[a, 'grade']
    for b in data4.index : 
        max_ = data4.loc[b,'max']
        max_lag = data4.loc[b,'max_lag']
        base = data4.loc[b,'base']
        gradient = data4.loc[b,'gradient']
            
        if val >max_lag and val <= max_ :
            if b not in [0, 51] : 
                data1.loc[a, 'grade_regul'] = base + gradient * (val - max_lag)
            elif b in [0] : 
                data1.loc[a, 'grade_regul'] = 0
            else : 
                data1.loc[a, 'grade_regul'] = 100
                

#%% 최종 데이터 셋 확인
data1['grade_regul'] = data1['grade_regul'].fillna(0)
data_fin = data1.sort_values(by = ['grade_regul'], axis = 0)


#%%
# 1) 51개의 cut에 맞는 분위수 값 할당
# 2) 각 컷으로 등급(GR)과 base할당. 이때 base는 2단위로 증가하고, 마지막 99.95 이상은 52로 할당하지 말고 51로 할당해야함
# 3) 등급별 최소값과 최대값 산출
# 4) 각 등급의 기울기 계산 후 점수 산식 적용
# In[ ]:
