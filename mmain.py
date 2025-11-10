import numpy as np
import config as conf
from ga import Ga
import matplotlib.pyplot as plt

config = conf.get_config()


def build_dist_mat(input_list):
    n = config.city_num
    dist_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = input_list[i, :] - input_list[j, :]
            # 计算点积
            dist_mat[i, j] = np.dot(d, d)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat

# 统一坐标，把钱换算成距离或者把距离换算成钱。
# 城市坐标加，删除
city_pos_list = np.array([[0.78572695 ,0.73535011],
 [0.21460802 ,0.63821997],
 [0.06759689 ,0.99245464],
 [0.53259774 ,0.24000999],
 [0.75682791 ,0.7002872 ],
 [0.53428582 ,0.50863317],
 [0.5492744  ,0.29674723],
 [0.20878283 ,0.88320611],
 [0.51767606 ,0.34844902],
 [0.01248232 ,0.45521919],
 [0.72482876 ,0.76727544],
 [0.50310398 ,0.2562581 ],
 [0.9605849  ,0.63935839],
 [0.64471631 ,0.93334495],
 [0.16638448 ,0.96336143]])
# 城市距离矩阵
city_dist_mat = build_dist_mat(city_pos_list)

print(city_pos_list)
print(city_dist_mat)

# 遗传算法运行
ga = Ga(city_dist_mat)
result_list, fitness_list = ga.train()
result = result_list[-1]
result_pos_list = city_pos_list[result, :]

# 绘图
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['KaiTi']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

fig = plt.figure()
plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-r')
plt.title(u"路线")
plt.legend()
fig.show()

fig = plt.figure()
plt.plot(fitness_list)
plt.title(u"适应度曲线")
plt.legend()
fig.show()
