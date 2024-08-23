import numpy as np
import scipy as sp
import scipy.sparse as spa
import methods as mt



# Daten laden
data=np.load("free_for_all_qpbenchmark-main/data/PEROLD.npz",allow_pickle=True)

# Geladenes Problem printen
lst = data.files
for item in lst:
    print(item)
    print(data[item])