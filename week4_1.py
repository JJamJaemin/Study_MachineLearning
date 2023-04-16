import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #표준점수 만들기위한 라이브러리
from sklearn.neighbors import KNeighborsClassifier

fish = pd.read_csv('https://bit.ly/fish_csv') #물고기 데이터 가져오기
fish.head()
print(pd.unique(fish['Species']))

fish_input = fish[['Weight','Length','Diagonal','Height','Width']].to_numpy() #5가지 정보 가져오기 numpy규격으로 바꾸기
print(fish_input[:5]) #5번째 까지 데이터 출력

fish_target = fish['Species'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state=42)
ss = StandardScaler() #표준점수화
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

kn = KNeighborsClassifier(n_neighbors=3) #확률을 다르게 할려면 숫자를 바꾼다.
kn.fit(train_scaled, train_target)

print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))