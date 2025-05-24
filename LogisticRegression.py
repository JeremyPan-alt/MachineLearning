import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iter=1000, lambda_reg=0.0):
        """
        初始化逻辑回归模型

        参数:
        learning_rate: 学习率
        max_iter: 最大迭代次数
        lambda_reg: 正则化强度
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """训练逻辑回归模型"""
        n_samples, n_features = X.shape

        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降优化
        for _ in range(self.max_iter):
            # 计算线性组合
            linear_model = np.dot(X, self.weights) + self.bias
            # 应用sigmoid函数
            y_pred = self.sigmoid(linear_model)

            # 计算梯度
            dw = (1 / n_samples) * (np.dot(X.T, (y_pred - y)) + self.lambda_reg * self.weights)
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 计算损失（用于监控训练过程）
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            loss += (self.lambda_reg / (2 * n_samples)) * np.sum(self.weights ** 2)
            self.loss_history.append(loss)

        return self

    def predict_proba(self, X):
        """预测样本属于正类的概率"""
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """预测样本类别"""
        y_pred_proba = self.predict_proba(X)
        return np.where(y_pred_proba >= threshold, 1, 0)

    def score(self, X, y):
        """计算模型准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def generate_non_linear_separable_data(n_samples=200, random_state=42):
    """生成部分线性不可分的数据"""
    np.random.seed(random_state)

    # 生成两类数据，有部分重叠
    mean1 = np.array([-2, -2])
    mean2 = np.array([2, 2])
    cov = np.eye(2) * 3  # 增加方差，使数据更分散

    # 生成样本
    X1 = np.random.multivariate_normal(mean1, cov, n_samples // 2)
    X2 = np.random.multivariate_normal(mean2, cov, n_samples // 2)
    X = np.vstack((X1, X2))

    # 生成标签
    y = np.hstack((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    # 打乱数据
    indices = np.random.permutation(n_samples)
    return X[indices], y[indices]


def plot_decision_boundary(X, y, model, title="逻辑回归决策边界"):
    """可视化决策边界"""
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.family'] = 'KaiTi'
    plt.rcParams['axes.unicode_minus'] = False

    # 设置颜色映射
    cmap = ListedColormap(['#FF0000', '#0000FF'])

    # 绘制散点图
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)

    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.title(title)
    plt.xlabel('特征 1')
    plt.ylabel('特征 2')

    plt.show()


def plot_loss_history(model):
    """可视化训练损失历史"""
    plt.figure(figsize=(10, 6))
    plt.rcParams['font.family'] = 'KaiTi'
    plt.plot(range(1, len(model.loss_history) + 1), model.loss_history)
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.title('训练过程中的损失变化')
    plt.grid(True)
    plt.show()


def main():
    # 生成部分线性不可分的数据
    X, y = generate_non_linear_separable_data(n_samples=300)

    # 划分训练集和测试集
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # 训练逻辑回归模型
    model = LogisticRegression(learning_rate=0.01, max_iter=1000, lambda_reg=0.1)
    model.fit(X_train, y_train)

    # 评估模型
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 可视化决策边界
    plot_decision_boundary(X, y, model, "逻辑回归分类器的决策边界")

    # 可视化损失历史
    plot_loss_history(model)


if __name__ == "__main__":
    main()