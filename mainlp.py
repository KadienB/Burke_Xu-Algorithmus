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
A_ub = data["A_ub"]
b_ub = data["b_ub"]
bounds = np.squeeze(data["bounds"])

# Rang der Gleichheits-Constraints-Matrix berechnen
rank_A_eq = np.linalg.matrix_rank(A_eq)
print(f"Rang von A_eq: {rank_A_eq}, Anzahl der Zeilen: {A_eq.shape[0]}")

# Überprüfen, ob A_eq vollen Zeilenrang hat
if rank_A_eq == A_eq.shape[0]:
    print("A_eq hat vollen Zeilenrang.")
else:
    print("A_eq hat keinen vollen Zeilenrang.")

# Rang der Ungleichheits-Constraints-Matrix berechnen
rank_A_ub = np.linalg.matrix_rank(A_ub)
print(f"Rang von A_ub: {rank_A_ub}, Anzahl der Zeilen: {A_ub.shape[0]}")

# Überprüfen, ob A_ub vollen Zeilenrang hat
if rank_A_ub == A_ub.shape[0]:
    print("A_ub hat vollen Zeilenrang.")
else:
    print("A_ub hat keinen vollen Zeilenrang.")

nonzeros_per_column = np.count_nonzero(A_eq, axis=0)

print("Matrix:")
print(A_eq)
print("\nAnzahl der Nicht-Null-Werte in jeder Spalte:")
np.set_printoptions(threshold=np.inf)
print(nonzeros_per_column)

# options = {"presolve": True, "maxiter": 0}
# result = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, options=options, method="highs")

# print(result)

