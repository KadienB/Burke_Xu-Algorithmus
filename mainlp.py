import numpy as np
import scipy as sp
import scipy.sparse as spa
import methods as mt
import burke_xu_lp as lp


""" Einstellungen """

test_case = 0
verbose = False
np.set_printoptions(precision=2, suppress=True, linewidth=400)


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

if test_case == 0:

    """ Laden der Daten """

    # Speichern der .npz Datei im Dictionary "data"
    data=np.load("free_for_all_qpbenchmark-main/data/CZPROB.npz", allow_pickle=True)

    # Auslesen der Daten aus dem Dictionary
    c = data["c"]
    A_eq = data["A_eq"]
    b_eq = data["b_eq"]
    A_ineq = data["A_ub"]
    b_ineq = data["b_ub"]
    bounds = np.squeeze(data["bounds"])
    A_std, b_std, c_std, transformations, sol_length, use_sparse = mt.lp_to_standardform(c=c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, verbose=verbose)
    """ Anwendung von scipy.optimize.linprog auf das Ausgangsproblem """

    # Lösung mit Linprog
    result1 = sp.optimize.linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    print(result1.x)
    print(np.dot(c, result1.x))


    """ Anwendung von scipy.optimize.linprog auf die Standardform des Problems """

    # Lösung mit Linprog in Standardform
    result2 = sp.optimize.linprog(c_std, A_eq = A_std, b_eq = b_std, method='highs')
    print(result2.x)
    print(A_std @ result2.x - b_std)
    print(np.dot(c_std, result2.x))

    # Zurückkonvertierte Lösung
    res, slack = mt.standardform_to_lp(x_std=result2.x, transformations=transformations, initial_length=sol_length, verbose=verbose)
    print(res)
    print(np.dot(c, res))


    """ Anwendung von burke_xu_lp auf die Standardform des Problems """

    # Lösung mit burke_xu_lp in Standardform
    x = lp.burke_xu_lp(c=c_std, A_eq=A_std, b_eq=b_std, verbose=verbose)
    result3 = mt.standardform_to_lp(x_std = x, transformations=transformations, initial_length=sol_length, verbose=verbose)
    print(result3)


    """ Anwendung von burke_xu_lp auf das Ausgangsproblem """

    # to be implemented




    """ Selbstgeschrieben Testcases """

# test von burke_xu_lp für übersichtliche Probleme
elif test_case == 1:

    c = np.array([-1, -2, 2])
    A_eq = np.array([[1, 1, 3],
                    [1, 2, 4]])
    b_eq = np.array([2, 3])

    # Lösen des linearen Programms
    result = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    my_result = lp.burke_xu_lp(c = c, A_eq = A_eq, b_eq = b_eq, maxiter=25, acc=1e-4, verbose=verbose)
    print(f"my_result = {my_result}")

    # Ausgabe der Ergebnisse
    print("Lösungsstatus:", result.message)
    print("Optimale Lösung x:", result.x)
    print("Optimaler Zielfunktionswert:", result.fun)

# test verschiedener Box-Restriktionen
if test_case == 2:

    c = np.array([-1, -2, -3])
    A_eq = np.array([[-1, 14, 3]])
    b_eq = np.array([7])
    A_ineq = np.array([[2, 3.5, -1], [1, 1, 11.8]])
    b_ineq = np.array([70, 50])
    # lb = np.array([-1, None, 0])
    # up = np.array([13, 7, 11])
    lb = np.array([-20, -10, -5])
    ub = np.array([15, 25, 30])
    bounds = np.vstack((lb, ub)).T

    print(f"c = {c}")
    print(f"A_eq = {A_eq}")
    print(f"b_eq = {b_eq}")
    print(f"A_ineq = {A_ineq}")
    print(f"b_ineq = {b_ineq} len(b_ineq) = {len(b_ineq)}")
    print(f"lb = {lb}")
    print(f"up = {ub}")
    print(f"bounds = {bounds}")

    # A_eq = spa.csc_matrix(A_eq)
    # A_ineq = spa.csc_matrix(A_ineq)

    A_std, b_std, c_std, transformations, sol_length, use_sparse = mt.lp_to_standardform(c=c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, verbose=True)

    print(np.linalg.matrix_rank(A_std))

    if use_sparse is False:
        print(f"A = {A_std}")
    elif use_sparse is True:
        print(f"A = {A_std.toarray()}")
    print(f"b = {b_std} len(b) = {len(b_std)}")
    print(f"c = {c_std}")
    print(f"transformations = {transformations}")
    print(f"sol_length = {sol_length}")
    print(f"use_sparse = {use_sparse}")

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