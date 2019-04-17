from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



pd.set_option('display.expand_frame_repr', False) # 출력값 요약 말고 다보이게 하는 코드 

# 많은 타입의 classifier 분류자 - 인공신경망, support vector machine, lions, tigers, bears, oh my
# decision tree 를 하는 이유 - 시각화 되어 이해하기 쉽기떄문     


# 1. import dataset. 2. train classifier 3. predict label for new flower. 


 
filename = 'train.csv' # 트레인 파일 가져오기
atable = pd.read_csv(filename, encoding='utf-8') # 읽기

print(atable)


test_data = pd.read_csv('test.csv', encoding='utf-8') # 읽기
test_data = test_data.drop('Name',axis=1) # 불필요하다고 생각하는 변수 제거하기 
test_data = test_data.drop('Cabin',axis=1)
test_data = test_data.drop('Fare',axis=1)
test_data = test_data.drop('Ticket',axis=1)
test_data = test_data.drop('SibSp',axis=1)
test_data = test_data.drop('Parch',axis=1) 
test_data = test_data.drop('PassengerId',axis=1)
test_data = test_data.drop('Embarked',axis=1)

test_data = test_data.fillna({'Age':30}) # 결측치 평균값 넣기 

test_data.loc[test_data["Sex"] == "male", "Sex"] = 0 # 값 숫자로 바꾸기 
test_data.loc[test_data["Sex"] == "female", "Sex"] = 1

print(test_data)

# combine =[atable] # 리스트에 담기 
# for dataset in combine :
#     dataset['Title'] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False) # 정규표현식
# print( pd.crosstab(atable['Title'], atable['Sex']))    

# print(atable) #[891 rows x 11 columns]
# print(type(atable)

# 차트 그리기
def bar_chart(feature):
    
    survived = atable[atable['Survived']==1][feature].value_counts()
    dead = atable[atable['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar',stacked=True, figsize=(10,5))
    plt.show() # %matplotlib inline 이거 이걸로 바꿔야함 
    
# bar_chart('Sex') # 여자가 더 많이 생존 / 의미있는 변수
# bar_chart('Pclass') # 1등급이 더 많이 생존 3등급이 더 많이 사망 의미있는 변수 
# bar_chart('Age') # 나이는 이방식으로 불가 전처리 해야함 
# bar_chart('SibSp') # 0과 1이 가장 많이 차지 의미있는 변수
# bar_chart('Parch') # 0과 1이 가장 많이 차지 의미있는 변수
# bar_chart('Embarked') #

corr = atable.corr()
print(corr)
#              PassengerId  Survived    Pclass       Age     SibSp     Parch      Fare
# PassengerId     1.000000 -0.005007 -0.035144  0.036847 -0.057527 -0.001652  0.012658
# Survived       -0.005007  1.000000 -0.338481 -0.077221 -0.035322  0.081629  0.257307
# Pclass         -0.035144 -0.338481  1.000000 -0.369226  0.083081  0.018443 -0.549500
# Age             0.036847 -0.077221 -0.369226  1.000000 -0.308247 -0.189119  0.096067
# SibSp          -0.057527 -0.035322  0.083081 -0.308247  1.000000  0.414838  0.159651
# Parch          -0.001652  0.081629  0.018443 -0.189119  0.414838  1.000000  0.216225
# Fare            0.012658  0.257307 -0.549500  0.096067  0.159651  0.216225  1.000000
#
# 지금 상황으로는 파악하기 어려움 


# 변수 이해하기 
# PassengerId - 걍 아이디 분석적의미없음 / pclass - 승선권 1, 2, 3 등급으로 나뉨
# survive - 0이면 죽음  1이면 생존  / sex - 문자로 되있기에 남자 여자를 0 1로 전처리 필요
# age 나이 / SibSp - 동반한 형제자매, 배우자 수에 따른 생존 여부
# Parch - 부모와 자녀수에 따른 생존여부
# ticket - 티켓 가격? 정확히 모르겠다. / Fare - 요금 / 
# embarked 승선한 위치 



# 1. Age의 약 20프의 데이터가 Null로 되어있다.
# 2. Cabin의 대부분 값은 Null이다.
# 3. Name, Sex, Ticket, Cabin, Embarked는 숫자가 아닌 문자 값이다.
#    - 연관성 없는 데이터는 삭제하거나 숫자로 바꿀 예정입니다.
#      (머신러닝은 숫자를 인식하기 때문입니다.)

# 의미업다고 생각하는 변수들 / PassengerId, Name,Ticket,Fare,Cabin,SibSp,Parch
# 결측치 처리 - 나이 값입력

# atable= atable.dropna( how ='any') # 결측치 제거


# 불필요하다고 생각하는 변수들 빼기
raw_data = atable.drop('Survived',axis=1) #서바이벌 변수 따로 뺌
raw_data = raw_data.drop('Name',axis=1)
raw_data = raw_data.drop('Cabin',axis=1)
raw_data = raw_data.drop('Fare',axis=1)
raw_data = raw_data.drop('Ticket',axis=1)
raw_data = raw_data.drop('SibSp',axis=1)
raw_data = raw_data.drop('Parch',axis=1) 
raw_data = raw_data.drop('PassengerId',axis=1)
raw_data = raw_data.drop('Embarked',axis=1)

#raw_data.fillna(raw_data.mean()) 
# raw_data.fillna(value=5.0)
# raw_data.dropna( how ='any') # 결측치는 다 뺴주는 것 
 
raw_data.loc[raw_data["Sex"] == "male", "Sex"] = 0 
raw_data.loc[raw_data["Sex"] == "female", "Sex"] = 1
 
# S는 Southhamton, Q는 queenstown,C는  Cherbourg / S가 대부분일기에 결측치 2개를 S로 넣어줌 
# raw_data = raw_data.fillna({'Embarked':'S'}) 
# raw_data.loc[raw_data["Embarked"] == "S", "Embarked"] = 0 
# raw_data.loc[raw_data["Embarked"] == "Q", "Embarked"] = 1
# raw_data.loc[raw_data["Embarked"] == "C", "Embarked"] = 2

print(raw_data.describe(include='all')) 
raw_data.shape
# print('분리전데이터',raw_data)
# raw_data =raw_data.dropna( how ='any') # 결측치는 다 뺴주는 것 
raw_data = raw_data.fillna({'Age':30}) 
print(raw_data.describe(include='all'))
# age 결측치는 어떻게 할것인가



raw_target = atable['Survived']

print(raw_data.shape, raw_target.shape)

# print('서바이벌데이터셋',train_target)
# atable[['Survived']]  atable['Survived'] 차이 숙지하기 

# train_set = atable[1:3] #  행기준 1부터 3까지 보여주는 
# print('트레인',train_set)

# raw_data.info()
# 
# #.2. train classifier
# # 나누기 
# 
# train_data= raw_data[0:100]
# train_target= raw_target[0:100]
# 
# print('트레인데이터',train_data)
# print('트레인타겟',train_target)
# 
# test_data= raw_data[100:]
# test_target= raw_target[100:]
# 
# print('테스트데이터',test_data)
# print('테스트타겟',test_target)
# #suv=atable.columns[0:1].tolist()
# # print(atable.shape)
# 
# clf = tree.DecisionTreeClassifier()
# clf.fit(train_data, train_target)
# print('테스트 타겟2',test_target) 
# print(clf.predict(test_data))

print(raw_data.info())


from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

clf = RandomForestClassifier(n_estimators=13)
clf

scoring = 'accuracy'

score = cross_val_score(clf, raw_data, raw_target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)

print(round(np.mean(score)*100, 2))

my_classifier = KNeighborsClassifier() 
my_classifier.fit(raw_data, raw_target )
# prediction =clf.predict(test_data)

predictions = my_classifier.predict(test_data)
print(predictions)

from sklearn.metrics import accuracy_score
# print(accuracy_score(Y_test, predictions))

acc_knn = round(my_classifier.score(raw_data, raw_target) * 100, 2)
print(acc_knn)