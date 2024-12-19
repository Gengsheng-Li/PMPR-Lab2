import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_3d_data():
    # 第一类数据
    class1_data = np.array([
        [ 1.58, 2.32, -5.8], [ 0.67, 1.58, -4.78], [ 1.04, 1.01, -3.63],
        [-1.49, 2.18, -3.39], [-0.41, 1.21, -4.73], [1.39, 3.16, 2.87],
        [ 1.20, 1.40, -1.89], [-0.92, 1.44, -3.22], [ 0.45, 1.33, -4.38],
        [-0.76, 0.84, -1.96]
    ])

    # 第二类数据
    class2_data = np.array([
        [ 0.21, 0.03, -2.21], [ 0.37, 0.28, -1.8], [ 0.18, 1.22, 0.16],
        [-0.24, 0.93, -1.01], [-1.18, 0.39, -0.39], [0.74, 0.96, -1.16],
        [-0.38, 1.94, -0.48], [0.02, 0.72, -0.17], [ 0.44, 1.31, -0.14],
        [ 0.46, 1.49, 0.68]
    ])

    # 第三类数据
    class3_data = np.array([
        [-1.54, 1.17, 0.64], [5.41, 3.45, -1.33], [ 1.55, 0.99, 2.69],
        [1.86, 3.19, 1.51], [1.68, 1.79, -0.87], [3.51, -0.22, -1.39],
        [1.40, -0.44, -0.92], [0.44, 0.83, 1.97], [ 0.25, 0.68, -0.99],
        [ 0.66, -0.45, 0.08]
    ])

    # 创建3D图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制散点图
    ax.scatter(class1_data[:, 0], class1_data[:, 1], class1_data[:, 2], 
              c='red', marker='o', label='Class 1', s=100)
    ax.scatter(class2_data[:, 0], class2_data[:, 1], class2_data[:, 2], 
              c='blue', marker='^', label='Class 2', s=100)
    ax.scatter(class3_data[:, 0], class3_data[:, 1], class3_data[:, 2], 
              c='green', marker='s', label='Class 3', s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.title('Three-Dimensional Data Distribution')
    ax.legend()
    ax.grid(True)

    # 计算并打印每类数据的统计信息
    for i, data in enumerate([class1_data, class2_data, class3_data], 1):
        print(f"\n第{i}类数据统计特征：")
        print(f"均值：{np.mean(data, axis=0)}")
        print(f"标准差：{np.std(data, axis=0)}")
        print(f"最大值：{np.max(data, axis=0)}")
        print(f"最小值：{np.min(data, axis=0)}")

    mean1 = np.mean(class1_data, axis=0)
    mean2 = np.mean(class2_data, axis=0)
    mean3 = np.mean(class3_data, axis=0)

    dist12 = np.linalg.norm(mean1 - mean2)
    dist13 = np.linalg.norm(mean1 - mean3)
    dist23 = np.linalg.norm(mean2 - mean3)

    print("\n类间欧氏距离：")
    print(f"第1类和第2类之间的距离：{dist12:.2f}")
    print(f"第1类和第3类之间的距离：{dist13:.2f}")
    print(f"第2类和第3类之间的距离：{dist23:.2f}")

    plt.show()

if __name__ == "__main__":
    visualize_3d_data()