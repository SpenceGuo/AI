import numpy as np

# 使用numpy生成数组
t1 = np.array([1, 2, 3, ])
print(t1)
print(type(t1))

t2 = np.arange(1, 10, 2)
print(t2)
print(type(t2))
# 查看数组内数据类型
print(t2.dtype)

t3 = np.array([1, 2, 3], dtype=np.float)
print(t3)
print(t3.dtype)

print('***********************************************')
t4 = np.arange(27).reshape((3, 3, 3))
print(t4)
