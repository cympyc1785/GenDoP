import numpy as np

src_path = "/data1/cympyc1785/caption/GenDoP/dataset/DATA/1_0000/shot_0070_depth.npy"

data = np.load(src_path)

print(data.shape)