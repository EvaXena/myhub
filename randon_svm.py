
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

def run_svm_demo():
    # 1. 生成模拟数据
    # 生成 40 个点，分为 2 类 (centers=2)，随机种子设为 6 以保证数据比较清晰
    X, y = make_blobs(n_samples=40, centers=2, random_state=0)

    # 2. 创建 SVM 模型并训练
    # kernel='linear' 表示我们要画一条直线（线性核）
    # C=1000 是惩罚系数，较大的 C 表示容忍更少的分类错误（即硬间隔）
    clf = svm.SVC(kernel='rbf', C=100)
    clf.fit(X, y)

    # --- 以下代码是为了可视化结果 (画图) ---

    # 创建图形
    plt.figure(figsize=(8, 6))
    
    # 画出数据点 (c=y 表示按类别上色, cmap是配色方案)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap=plt.cm.Paired, label='Data Points')

    # 获取当前的坐标轴范围
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # 创建网格来评估模型
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    
    # 计算网格中每个点到超平面的距离 (decision_function)
    Z = clf.decision_function(xy).reshape(XX.shape)

    # 画出边界和间隔
    # levels=[-1, 0, 1] 分别代表：下边界、超平面(决策边界)、上边界
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # 圈出支持向量 (Support Vectors)
    # clf.support_vectors_ 存储了所有支持向量的坐标
    ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=150,
               linewidth=1.5, facecolors='none', edgecolors='k', label='Support Vectors')

    plt.title("SVM Linear Classification Demo")
    plt.legend(loc="upper right")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

    # 3. 简单的预测演示
    new_point = [[7, -6.5]]  # 假设有一个新点在坐标 (0,0)
    prediction = clf.predict(new_point)
    print(f"模型训练完成。")
    print(f"新数据点 {new_point} 的分类结果是: 类别 {prediction[0]}")
    print(f"在这个模型中，一共有 {len(clf.support_vectors_)} 个支持向量起到了决定性作用。")

if __name__ == "__main__":
    run_svm_demo()