import os
import time
import numpy as np
import scipy as sp
import scipy.sparse as spa
import sksparse as skit
import methods as mt
import burke_xu_lp as lp
from memory_profiler import profile


""" Einstellungen """

loop = False
test_case = 0
verbose = True
acc = 1e-4
maxiter = 100
# np.set_printoptions(threshold=np.inf)
# np.set_printoptions(precision=2, suppress=True, linewidth=400)

if loop == True:
    # Pfad zu den .npz-Dateien
    data_path = "/workspaces/Python/free_for_all_qpbenchmark-main/data"
    output_file = os.path.join(data_path, "testergebnisse.txt")

    # Alle .npz-Dateien im Verzeichnis durchlaufen
    npz_files = [f for f in os.listdir(data_path) if f.endswith('.npz')]

    # Datei einmalig öffnen, Kopfzeile schreiben
    with open(output_file, "w") as file:
        file.write(f"{'Datei':<30}{'Fun':<20}{'Nullstep':<10}{'MaxIter':<10}{'Time':<15}{'Status':<10}{'Result':<50}{'Slack':<50}\n")
        file.write("="*180 + "\n")

    # Ergebnisse in die Textdatei schreiben, nach jedem Testfall
    for npz_file in npz_files:
        try:
            # Laden der Daten
            data = np.load(os.path.join(data_path, npz_file), allow_pickle=True)
            c = data["c"]
            A_eq = spa.csc_matrix(data["A_eq"])
            b_eq = data["b_eq"]
            A_ineq = spa.csc_matrix(data["A_ub"])
            b_ineq = data["b_ub"]
            bounds = np.squeeze(data["bounds"])
            
            # Algorithmus 1
            start_time = time.time()
            try:
                result, slack, fun, nullstep, maxiter, exec_time = lp.burke_xu_lp(
                    c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, 
                    bounds=bounds, maxiter=maxiter, acc=acc, verbose=verbose
                )
                status = "Erfolg"
            except Exception as e:
                # Algorithmus 2 bei Fehler
                try:
                    result, slack, fun, nullstep, maxiter, exec_time = lp.burke_xu_lp(
                        c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, 
                        bounds=bounds, maxiter=100, acc=acc, regularizer=(acc*acc), verbose=verbose
                    )
                    status = "CrashErfolg"
                except Exception as e2:
                    status = "Fehler"
                    result, slack, fun, nullstep, maxiter, exec_time = None, None, None, None, None, None
            
            # Datei öffnen und schreiben
            with open(output_file, "a") as file:
                if status == "Erfolg" or status == "CrashErfolg":
                    file.write(f"{npz_file:<30}{fun:<20.10f}{nullstep:<10}{maxiter:<10}{exec_time:<15.6f}{status:<10}{str(result)[:50]:<50}{str(slack)[:50]:<50}\n")
                else:
                    file.write(f"{npz_file:<30}{'':<20}{'':<10}{'':<10}{'':<15}{status:<10}\n")
        
        except Exception as e:
            # Fehler beim Laden der Datei oder beim Ausführen des Algorithmus
            with open(output_file, "a") as file:
                file.write(f"{npz_file:<30}{'':<20}{'':<10}{'':<10}{'':<15}{'Fehler':<10}\n")

    print(f"Die Ergebnisse wurden in '{output_file}' gespeichert.")


if test_case == 0:

    """ Laden der Daten """

    # Speichern der .npz Datei im Dictionary "data"
    data=np.load("free_for_all_qpbenchmark-main/data/CZPROB.npz", allow_pickle=True)

    # Auslesen der Daten aus dem Dictionary
    c = data["c"]
    A_eq = spa.csc_matrix(data["A_eq"])
    b_eq = data["b_eq"]
    A_ineq = spa.csc_matrix(data["A_ub"])
    b_ineq = data["b_ub"]
    bounds = np.squeeze(data["bounds"])

    if verbose:
        print(f"c = {c}")
        print(f"A_eq = {A_eq}")
        print(f"b_eq = {b_eq}")
        print(f"A_ineq = {A_ineq}")
        print(f"b_ineq = {b_ineq}")
        print(f"bounds = {bounds}")

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
    result2back, slack = mt.standardform_to_lp(x_std=result2.x, transformations=transformations, initial_length=sol_length, verbose=verbose)
    print(result2back)
    print(np.dot(c, result2back))


    """ Anwendung von burke_xu_lp auf die Standardform des Problems """

    # Lösung mit burke_xu_lp in Standardform
    # x = lp.burke_xu_lp(c=c_std, A_eq=A_std, b_eq=b_std, maxiter=100, acc=1e-4, verbose=verbose)
    # result3 = mt.standardform_to_lp(x_std = x, transformations=transformations, initial_length=sol_length, verbose=verbose)
    # print(result3)
    # print(np.dot(c_std, result3))


    """ Anwendung von burke_xu_lp auf das Ausgangsproblem """

    # Lösung mit burke_xu_lp des eigentlichen Problems
    result3back = lp.burke_xu_lp(c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, maxiter=maxiter, acc=acc, verbose=verbose)
    print(result3back[0])
    print(np.dot(c, result3back[0]))

    print("----------------------------------------")
    print(result1.x)
    print(np.dot(c, result1.x))
    print(result2back)
    print(np.dot(c, result2back))


    """ Selbstgeschrieben Testcases """

# test von burke_xu_lp für übersichtliche Probleme
elif test_case == 1:

    c = np.array([-1, -2, 2])
    A_eq = np.array([[1, 1, 3],
                    [1, 2, 4]])
    b_eq = np.array([2, 3])

    A_eq = spa.csc_matrix(A_eq)

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