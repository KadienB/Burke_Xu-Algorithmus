import os
from typing import Iterator, Union

import numpy as np
import qpbenchmark
import scipy.io as spio
import scipy.sparse as spa
from qpbenchmark.benchmark import main
from qpbenchmark.problem import Problem
from qpsolvers import solve_qp

data=np.load("free_for_all_qpbenchmark-main/databackup/LIPMWALK0.npz",allow_pickle=True)
lst = data.files
# for item in lst:
#     print(item)
#     print(data[item])

# print(lst)
# P = data['P']
# q = data['q']
# G = data['G']
# h = data['h']

# x = solve_qp(P, q, G, h, solver="proxqp")
# print(f"QP solution: {x = }")

def BurkeXu(
        
        
) -> 