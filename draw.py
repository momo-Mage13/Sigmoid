"""
画图
包含 3 个函数：draw_loss()、draw_learning_rate()、draw_decision_boundary()
三个函数都只画图，不调用 plt.show()：main.py 负责保存到 output 文件夹，最后统一显示
"""
import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import Logistic_Regression


def draw_loss(loss_list):
    """
    函数 5　画迭代轮数与损失函数的关系图
    功能：横轴为迭代轮数，纵轴为损失函数值，画出损失随迭代轮数变化的曲线
    输入：loss_list：Logistic_Regression() 返回的损失列表
    输出：无（只画图，不调用 plt.show()）
    """
    plt.figure()                                                 # 新建一张图
    plt.plot(range(1, len(loss_list) + 1), loss_list, color='tab:blue')   # 横轴从第 1 轮开始
    plt.xlabel('迭代轮数')
    plt.ylabel('损失函数值')
    plt.title('迭代轮数与损失函数的关系（共 {} 轮）'.format(len(loss_list)))
    plt.grid(True, linestyle='--', alpha=0.5)


def draw_learning_rate(x, y, lr_list, num_iter):
    """
    函数 6　画不同学习率下损失函数的变化图
    功能：对 lr_list 中的每个学习率，调用 Logistic_Regression() 各训练 num_iter 轮
          （threshold 取 0，不提前停止），把各自的损失曲线画在同一张图上，用图例标出学习率
    输入：x、y：特征和标签
          lr_list：学习率列表
          num_iter：迭代轮数
    输出：无（只画图，不调用 plt.show()）
    """
    plt.figure()                                                 # 新建一张图，所有学习率画在一起
    for lr in lr_list:
        loss_list, _ = Logistic_Regression(x, y, lr, num_iter, 0)   # threshold 取 0，跑满 num_iter 轮
        plt.plot(range(1, len(loss_list) + 1), loss_list, label='学习率 {}'.format(lr))
    plt.xlabel('迭代轮数')
    plt.ylabel('损失函数值')
    plt.title('不同学习率下损失函数的变化')
    plt.legend()                                                 # 显示图例
    plt.grid(True, linestyle='--', alpha=0.5)


def draw_decision_boundary(x, y, w):
    """
    函数 7　画分类线
    功能：画出所有同学的散点图（挂科和没挂科用不同颜色、大小区分），
          再画出分类线 w0 + w1 × x1 + w2 × x2 = 0
    输入：x：特征，形状 (m, 2)，第 1 列学习时长，第 2 列刷手机时长
          y：标签，形状 (m,)
          w：Logistic_Regression() 返回的参数 [w0, w1, w2]
    输出：无（只画图，不调用 plt.show()）
    """
    plt.figure()                                                 # 新建一张图
    fail = (y == 1)                                              # 挂科的同学
    passed = (y == 0)                                            # 没挂科的同学

    # 散点图：横轴学习时长，纵轴刷手机时长；挂科红色大点，没挂科蓝色小点
    plt.scatter(x[passed, 0], x[passed, 1], c='tab:blue', s=25, alpha=0.6, label='没挂科')
    plt.scatter(x[fail, 0], x[fail, 1], c='tab:red', s=45, alpha=0.6, label='挂科')

    # 分类线：w0 + w1*x1 + w2*x2 = 0  →  x2 = -(w0 + w1*x1) / w2
    x1 = np.linspace(x[:, 0].min() - 1, x[:, 0].max() + 1, 100)   # 学习时长的取值范围
    x2 = -(w[0] + w[1] * x1) / w[2]
    plt.plot(x1, x2, color='red', linewidth=2, label='分类线')

    plt.xlabel('每周课后学习时长（小时）')
    plt.ylabel('每天刷手机娱乐时长（小时）')
    plt.title('期末挂科预警分类图')
    plt.xlim(x[:, 0].min() - 1, x[:, 0].max() + 1)
    plt.ylim(x[:, 1].min() - 1, x[:, 1].max() + 1)              # 纵轴按数据范围显示，避免分类线把图拉得过高
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
