import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #표준점수 만들기위한 라이브러리
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier



fish = pd.read_csv('https://bit.ly/fish_csv') #물고기 데이터 가져오기
fish.head() #첫 행을 출력하는 메서드
print(fish.head())

print(pd.unique(fish['Species']))

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy() #5가지 정보 가져오기 numpy규격으로 바꾸기
print(fish_input[:5]) #5번째 까지 데이터 출력
fish_target = fish['Species'].to_numpy()#fish_target생성

train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)#테스트데이터 나누기

ss = StandardScaler() #표준점수화
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

#####K최근접 이웃 사용#####
# kn = KNeighborsClassifier(n_neighbors=3) #확률을 다르게 할려면 숫자를 바꾼다.
# kn.fit(train_scaled, train_target)
#
# print("k최근접이웃 학습정확도",kn.score(train_scaled, train_target))
# print("k최근접이웃 테스트정확도",kn.score(test_scaled, test_target))
#
# print(kn.classes_)#kn.classes,란 저장된 정렬된 타겟값
# print(kn.predict(test_scaled[:5]))#테스트 데이터에서 첫5개의 예측한 값
#
# proba = kn.predict_proba(test_scaled[:5]) #predict_proba는 클래스별 확률값 반환
# print(np.round_(proba, decimals=4)) #소수점 4자리까지 표현
#
# distances, indexs = kn.kneighbors(test_scaled[3:4]) #위에서 확률 예측한게 맞는지 확인하기위해 꺼낸 데이터
# print(train_target[indexs]) #결과 확인

####################
#####로지스틱 회귀 사용#########
z = np.arange(-5, 5, 0.1)
phi = 1 / (1+np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show() #시그모이드 함수(로지스틱함수) 그리기

###로지스틱으로 이진분류 하기###
char_arr = np.array(['A','B','C','D','E'])
print(char_arr[[True, False, True, False, False]]) #이진분류할 도미와 빙어만 골라내기

#train_scaled와 train_target에서 도미와 빙어 데이터만 골라내기
bream_smelt_indexs = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexs]
target_bream_smelt = train_target[bream_smelt_indexs]

lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5])) #로지스틱으로 도미 빙어 이진분류 예측
print(lr.predict_proba(train_bream_smelt[:5])) #확률값 출력하기

print(lr.classes_) #양성(1) 음성(0) 확인하기 (타겟의 알파벳 순서로 정해짐)

print(lr.coef_, lr.intercept_) #로지스틱 식 계수 확인하기

decisions = lr.decision_function(train_bream_smelt[:5]) #5개 샘플의 시그모이드 함수인 Z값 구하기
print(decisions )

from scipy.special import expit
print(expit(decisions)) #Z값마다 양성 확률 구하기

####로지스틱으로 다중분류 하기####
lr = LogisticRegression(C=20, max_iter=1000) #C는 alpha처럼 규제의 양 조절 alph와 반대로 적용 클수록 규제 완화
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))

print(lr.predict(test_scaled[:5]))#테스트세트 예측 출력

proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3)) #예측에 대한 확률 출력 소수점 3자리까지

print(lr.classes_) #클래스 확인하기

####로지스틱 다중분류 일때 선형방정식 모습 구하기####
print(lr.coef_.shape, lr.intercept_.shape)

decision = lr.decision_function(test_scaled[:5])
print(np.round(decision,decimals=2)) #z1~z7까지의 값을 확률로 표현하기

from scipy.special import softmax #로지스틱의 다중분류일때는 시그모이드 함수가 아닌 소프트맥스함수
proba = softmax(decision, axis=1) #axis=1을 통해 각행마다 확률
print(np.round(proba, decimals=3))