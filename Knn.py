import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier # <--- 这里变了
from sklearn.datasets import make_blobs

def run_knn_demo():
    # 1. 生成数据 (和之前 SVM 一样)
    X, y = make_blobs(n_samples=50, centers=2, random_state=6, cluster_std=1.2)

    # 2. 创建 KNN 模型
    # n_neighbors=3 表示看最近的 3 个邻居
    clf = KNeighborsClassifier(n_neighbors=3)
    
    # 3. 训练
    # 注意：KNN 的 fit 其实几乎不耗时，它只是把数据存进内存里
    clf.fit(X, y)

    # --- 以下是画图代码 (为了看清楚边界的区别) ---
    plt.figure(figsize=(8, 6))
    
    # 画出背景网格的分类结果
    h = .02  # 网格步长
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格中每个点的类别
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 画出彩色区域
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3, shading='auto')

    # 画出训练数据点
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=50)

    plt.title("KNN Classification (K=3)")
    plt.show()

    # 4. 预测新数据
    new_point = [[0, 0]]
    print(f"新数据点 {new_point} 的分类结果: {clf.predict(new_point)[0]}")
    # 还可以看看最近的邻居是谁
    distances, indices = clf.kneighbors(new_point)
    print(f"最近的 {clf.n_neighbors} 个邻居的索引: {indices}")
    print(f"到邻居的距离: {distances}")

if __name__ == "__main__":
    run_knn_demo()