import pickle
import matplotlib.pyplot as plt
import numpy as np

with open("test3", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

plt.plot(b)
plt.show()

# a = np.array([[1,float("nan"),3],[float("nan"),5,6],[7,8,float("nan")]])
# print(a)
# b = a[~np.isnan(a)]
# print(b)