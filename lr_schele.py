import numpy as np
import matplotlib.pyplot as plt
# 定义多项式系数
coeffs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 创建一个二次多项式函数
# poly_func = np.poly1d(coeffs)

lr = 0.01
lr_schedule = lambda x : lr * (1 + x * (10. / 30.))
# 计算x=2时的函数值
# print(poly_func(2))

# x = np.linspace(0, 1, 100)
# y = poly_func(x)

x = np.linspace(0, 30, 100)
y = lr_schedule(x)
plt.plot(y, label='f_mean')
plt.savefig("lr_test.png")