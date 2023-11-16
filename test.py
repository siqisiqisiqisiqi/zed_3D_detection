import numpy as np


dt = np.dtype([('name', np.str_, 16), ('grades', np.float64, (2,))])
x = np.array([('Sarah', (8.0, 7.0)), ('John', (6.0, 7.0))], dtype=dt)
t = x[0]
print("test")

shape = np.array([[1,2,0,4],[2,3,0,0]])
logits = np.nonzero(shape < 1)
print(np.max(shape))
# print(shape[logits[0], logits[1]])


# test = np.array([float("nan"), 1, 2, 3, float("nan")])
# test2 = np.array([test,test,test])
# print(test2.shape)
# test3 = np.stack((test2,test2), axis=2)
# print(test3.shape)

# with np.load('params/E.npz') as X:
#     mtx, dist, Mat, tvecs = [X[i] for i in ("mtx", "dist", "Matrix", "tvec")]
# print(tvecs)
