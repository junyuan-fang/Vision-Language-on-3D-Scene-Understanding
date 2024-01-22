import numpy as np
import binvox_rw

with open('airplane_0001.binvox', 'rb') as f:
    m1 = binvox_rw.read_as_3d_array(f)
    data =m1.data
    print(data.shape)#(30,30,30)
print(type(data))

# A = np.array([[1,2],[3,4]])
# print(A.shape)
# B = np.repeat(A[np.newaxis, ...], 2, axis=0)
# print(B)
# print(B.shape)