import numpy as np
#
# dzdx = np.array([[1,2,3], [1,2,5], [4,5,6]])
# M = np.matrix([[1,2,3], [1,2,5], [4,5,6]])
#
# print(dzdx.shape==(3,3))
# print(M.shape)
# print(dzdx.size)
# print(M.transpose())
#
# print(M)
# print(M[:,1].transpose()[0])
# R=M
# print((R.shape == (3,3)) & (R.size == 9))
#
# AA = np.array([1,2,3,1])
# print(AA.size)
# print((AA.shape == (4,1)) | (AA.shape == (1,4)))
# print(AA)
# print(AA[0:2])
#
# # rows, cols = np.array(dzdx).shape
#
# # wx, wy = np.meshgrid(np.arange(-np.pi/2 , np.pi/2 , np.pi / (cols - 1)), np.arange(-np.pi/2, np.pi/2, np.pi/(rows - 1)))
#
# # print(np.arange(1,4,0.2))
#
# # print(wx)
# # print(wy)

omega = 1.4
theta = 0.4
print(np.identity(3) + omega * np.sin(theta) + omega * omega * (1 - np.cos(theta)))

a=[1,3,45]
print(a.append(3))

A = np.array([[1, 0, 1], [-1, -2, 0], [0, 1, -1]])
A = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
U, S_, V_t = np.linalg.svd(A, full_matrices=True)
V = V_t.T
print(U)
S = np.zeros((len(U), len(V_t)))
for idx, element in enumerate(S_):
    S[idx, idx] = element
print(S)
print(V)

print(U.dot(S).dot(V_t))

print(np.eye(3))
d = np.zeros((3,3))
a = np.fill_diagonal(d, np.NaN)
print(d)
