import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

#DataFrame ==> np ==> 표준화(평균을 중심으로 동일한 표준편차)

wine = pd.read_csv('wine.csv')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()

target = wine['class'].to_numpy()



train_input, test_input, train_target, test_target = train_test_split(

    data, target, test_size=0.2, random_state=42)



ss = StandardScaler()

ss.fit(train_input)

train_scaled = ss.transform(train_input)

test_scaled = ss.transform(test_input)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(10,7))
plot_tree(dt)
plt.show()

plt.figure(figsize=(10,7))
plot_tree(dt, max_depth=1, filled =True, feature_names=['alcohol','sugar','pH'])
plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)

print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True, feature_names=['alcohol','sugar','pH'])
plt.show()

dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)

print(dt.score(train_input,train_target))
print(dt.score(test_input, test_target))

plt.figure(figsize=(20,15))
plot_tree(dt, filled=True,feature_names=['alcohol','sugar','pH'])

plt.show()
print(dt.feature_importances_)

#검증
#교차검증
sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42
)
from sklearn.model_selection import cross_validate
import numpy as np
scores = cross_validate(dt, train_input, train_target)
print(scores)
print(np.mean(scores['test_score']))

#분할기를 사용한 교차검증
from sklearn.model_selection import StratifiedKFold

scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold)
print(np.mean(scores['test_score']))

splitter =  StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))

from sklearn.model_selection import GridSearchCV

params = {'min_impurity_decrease ': [0.0001,0.0002,0.0003,0.0004,0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1)
gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))

print(gs.best_params_)

print(gs.cv_results_['mean_test_score'])

from scipy.stats import uniform, randint
rgen = randint(0,10)
r