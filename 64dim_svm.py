import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

def run_high_dim_svm():
    # 1. 加载数据：手写数字图片 (8x8 像素)
    digits = datasets.load_digits()

    # 让我们看看数据长什么样
    # images 是 8x8 的原始图片，data 是铺平后的 64维 数组
    print(f"原始图片形状: {digits.images.shape}")  # (1797, 8, 8)
    print(f"输入特征维度: {digits.data.shape}")    # (1797, 64) -> 这就是64维！

    # 2. 数据准备
    n_samples = len(digits.images)
    data = digits.data # 已经扁平化了，直接用

    # 拆分训练集 (80%) 和测试集 (20%)
    # shuffle=False 是为了让结果可复现，实际中通常设为 True
    X_train, X_test, y_train, y_test = train_test_split(
        data, digits.target, test_size=0.2, shuffle=False
    )

    # 3. 创建并训练 SVM 模型
    # gamma=0.001 是 RBF 核的一个参数，对于这种像素数据很关键
    clf = svm.SVC(gamma=0.001) 
    
    print(f"正在 64 维空间中训练 SVM，请稍候...")
    clf.fit(X_train, y_train)

    # 4. 预测
    predicted = clf.predict(X_test)

    # 5. 评估结果
    print(f"\n模型准确率: {metrics.accuracy_score(y_test, predicted):.2%}")
    print("--------------------------------------------------")
    
    # --- 可视化：展示前 4 个预测结果 ---
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        # 把 64维 数据还原回 8x8 图片以便人类观看
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Pred: {prediction}")

    plt.suptitle("SVM Prediction on Handwritten Digits (64 Dimensions)")
    plt.show()

    # 看看如果有一张稍微复杂的图，它具体把哪些混淆了
    # 打印分类报告
    print(f"分类详细报告:\n{metrics.classification_report(y_test, predicted)}")

if __name__ == "__main__":
    run_high_dim_svm()