import numpy as np
import scipy as sp
import scipy.sparse as spa
import methods as mt
from burke_xu_lp import burke_xu_lp



# Daten laden
data=np.load("free_for_all_qpbenchmark-main/data/CZPROB.npz", allow_pickle=True)
c = data["c"]
A_eq = data["A_eq"]
b_eq = data["b_eq"]
A_ineq = data["A_ub"]
b_ineq = data["b_ub"]
bounds = np.squeeze(data["bounds"])

# # Rang der Gleichheits-Constraints-Matrix berechnen
# rank_A_eq = np.linalg.matrix_rank(A_eq)
# print(f"Rang von A_eq: {rank_A_eq}, Anzahl der Zeilen: {A_eq.shape[0]}")

# # Überprüfen, ob A_eq vollen Zeilenrang hat
# if rank_A_eq == A_eq.shape[0]:
#     print("A_eq hat vollen Zeilenrang.")
# else:
#     print("A_eq hat keinen vollen Zeilenrang.")

# # Rang der Ungleichheits-Constraints-Matrix berechnen
# rank_A_ub = np.linalg.matrix_rank(A_ub)
# print(f"Rang von A_ub: {rank_A_ub}, Anzahl der Zeilen: {A_ub.shape[0]}")

# # Überprüfen, ob A_ub vollen Zeilenrang hat
# if rank_A_ub == A_ub.shape[0]:
#     print("A_ub hat vollen Zeilenrang.")
# else:
#     print("A_ub hat keinen vollen Zeilenrang.")

# nonzeros_per_column = np.count_nonzero(A_eq, axis=0)

# print("Matrix:")
# print(A_eq)
# print("\nAnzahl der Nicht-Null-Werte in jeder Spalte:")
# np.set_printoptions(threshold=np.inf)
# print(nonzeros_per_column)

# options = {"presolve": True, "maxiter": 0}
# result = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=options, method="highs")

# print(result)
A_std, b_std, c_std, transformations, sol_length, use_sparse = mt.lp_to_standardform(c=c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, verbose=True)

result1 = sp.optimize.linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
print(result1.x)
print(np.dot(c, result1.x))
result2 = sp.optimize.linprog(c_std, A_eq = A_std, b_eq = b_std, method='highs')
print(result2.x)
print(A_std @ result2.x - b_std)
print(np.dot(c_std, result2.x))
res, slack = mt.standardform_to_lp(x_std=result2.x, transformations=transformations, initial_length=sol_length, verbose=True)
print(res)
print(np.dot(c, res))

x = burke_xu_lp(c=c_std, A_eq=A_std, b_eq=b_std, verbose=False)
result3 = mt.standardform_to_lp(x_std = x, transformations=transformations, initial_length=sol_length, verbose=False)
print(result3)
