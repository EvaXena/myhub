from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. 准备数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 定义我们要尝试的参数组合
# 就像是一个菜单，让电脑每一种搭配都尝一口
param_grid = [
    # 第一组尝试：用线性核，只调 C
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
    
    # 第二组尝试：用 RBF 核，同时调 C 和 gamma
    {'kernel': ['rbf'], 'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001]}
]

# 3. 创建 SVM 模型（此时是个空壳，不带参数）
svc = svm.SVC()

# 4. 创建网格搜索对象
# cv=5 表示用 5折交叉验证（数据轮流做验证集，保证结果靠谱）
clf = GridSearchCV(svc, param_grid, cv=5, scoring='accuracy', verbose=1)

# 5. 开始跑数据 (这一步最耗时)
print("开始自动调参...")
clf.fit(X_train, y_train)

# 6. 输出结果
print("\n----------------------------------------")
print(f"找到的最好参数是: {clf.best_params_}")
print(f"最好的验证集分数: {clf.best_score_:.4f}")
print("----------------------------------------")

# 7. 用最好的模型去预测测试集
best_model = clf.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"最终测试集准确率: {accuracy:.2%}")