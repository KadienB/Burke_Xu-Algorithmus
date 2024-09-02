#!/usr/bin/env python3
import os
import time
import numpy as np
import scipy as sp
import scipy.sparse as spa
import sksparse as skit
import pandas as pd
import methods as mt
import burke_xu_lp as lp
from memory_profiler import profile


""" Einstellungen """

loop = True
test_case = -1
verbose = False
acc = 1e-4
maxiter = 120
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
        file.write(f"{'Datei':<20}{'Fun':<15}{'Nullstep':<10}{'IterCount':<10}{'Time':<15}{'Status':<10}{'Result':<50}{'Slack':<50}\n")
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
            
            # Standardausführrung des Algorithmus
            start_time = time.time()
            try:
                result, slack, fun, nullstep, iter, exec_time = lp.burke_xu_lp(
                    c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, 
                    bounds=bounds, maxiter=maxiter, acc=acc, verbose=verbose
                )
                status = "Erfolg"
            except Exception as e:
                # Tritt ein Fehler auf: Ausführung des Algurithmus mit Regularisierung
                try:
                    result, slack, fun, nullstep, iter, exec_time = lp.burke_xu_lp(
                        c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, 
                        bounds=bounds, maxiter=50, acc=1e-3, regularizer=1e-6, verbose=verbose
                    )
                    status = "CrashErfolg"
                except Exception as e2:
                    status = "Fehler"
                    result, slack, fun, nullstep, iter, exec_time = None, None, None, None, None, None
            
            # Datei öffnen und schreiben
            with open(output_file, "a") as file:
                if status == "Erfolg" or status == "CrashErfolg":
                    file.write(f"{npz_file:<20}{fun:<15.4E}{nullstep:<10}{iter:<10}{exec_time:<15.6f}{status:<10}{str(result)[:50]:<50}{str(slack)[:50]:<50}\n")
                else:
                    file.write(f"{npz_file:<20}{'':<15}{'':<10}{'':<10}{'':<15}{status:<10}\n")
        
        except Exception as e:
            # Fehler beim Laden der Datei oder beim Ausführen des Algorithmus
            with open(output_file, "a") as file:
                file.write(f"{npz_file:<30}{'':<20}{'':<10}{'':<10}{'':<15}{'Fehler':<10}\n")

    print(f"Die Ergebnisse wurden in '{output_file}' gespeichert.")


if test_case == 0:

    """ Laden der Daten """

    # Speichern der .npz Datei im Dictionary "data"
    filepath = "free_for_all_qpbenchmark-main/data/CZPROB.npz"
    data=np.load(filepath, allow_pickle=True)

    # Auslesen der Daten aus dem Dictionary
    c = data["c"]
    A_eq = spa.csc_matrix(data["A_eq"])
    b_eq = data["b_eq"]
    A_ineq = spa.csc_matrix(data["A_ub"])
    b_ineq = data["b_ub"]
    bounds = np.squeeze(data["bounds"])

    if verbose:
        print(f"c = {c}")
        print(f"A_eq = {A_eq.toarray()}")
        print(f"b_eq = {b_eq}")
        print(f"A_ineq = {A_ineq.toarray()}")
        print(f"b_ineq = {b_ineq}")
        print(f"bounds = {bounds}")

    A_std, b_std, c_std, transformations, sol_length, use_sparse = mt.lp_to_standardform(
        c=c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, verbose=verbose)
    
    
    # Anwendung von scipy.optimize.linprog auf das Ausgangsproblem
    result1 = sp.optimize.linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    print(result1.x)
    print(np.dot(c, result1.x))


    # Anwendung von scipy.optimize.linprog auf die Standardform des Problems
    result2 = sp.optimize.linprog(c_std, A_eq = A_std, b_eq = b_std, method='highs')
    print(result2.x)
    print(np.dot(c_std, result2.x))

    # Zurückkonvertierte Lösung
    result2back, slack = mt.standardform_to_lp(x_std=result2.x, transformations=transformations, initial_length=sol_length, verbose=verbose)
    print(result2back)
    print(np.dot(c, result2back))


    # Anwendung von burke_xu_lp auf das Ausgangsproblem
    result3back = lp.burke_xu_lp(c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, maxiter=maxiter, acc=acc, verbose=verbose)
    
    print(f"A_eq hatte die Form")
    print(f"{pd.DataFrame(A_eq.toarray())}")
    print(f"b_eq hatte die Form {b_eq}.")
    print(f"A_ineq hatte die Form")
    print(f"{pd.DataFrame(A_ineq.toarray())}")
    print(f"b_ineq hatte die Form {b_ineq}.")
    print(f"bounds hat die Form {bounds}.")
    print(f"Die in Standardform übeführte Matrix hatte die Form")
    print(f"{pd.DataFrame(A_std.toarray())}")
    print(f"MatrixName(Rang : Zeilen) = A_std({np.linalg.matrix_rank(A_std.toarray())} : {A_std.shape[0]}), A_eq({np.linalg.matrix_rank(A_eq.toarray())} : {A_eq.shape[0]}), A_ineq({np.linalg.matrix_rank(A_ineq.toarray())} : {A_ineq.shape[0]}).")
    print(f"Der Lösungsvektor lautet {result3back[0]}.")
    print(f"Der Minimale Funktionswert lautet {np.dot(c, result3back[0])} = {result3back[2]} mit Genauigkeit {acc:.0e} bei Datei {os.path.splitext(os.path.basename(filepath))[0]} aus Netlib.")
    print(f"Es wurde {result3back[3]} mal der Nullstep verwendet, also der Prädiktor-Schritt abgelehnt.")
    print(f"Insgesamt wurden {result3back[4]} Schritte verwendet, wobei {maxiter} die maximale Anzahl der Schritte war.")
    print(f"Es wurden {result3back[5]} Sekunden benötigt.")
    
    print("----------------------------------------")

    print("Linprog auf das Ausgangsproblem ergab:")
    print(result1.x)
    print(np.dot(c, result1.x))
    print("Linprog auf das Standardproblem ergab:")
    print(result2back)
    print(np.dot(c, result2back))


    """ ---------------------------------------------------------------------------------------------------------------------- """
    """ Selbstgeschrieben Testcases """
    """ ---------------------------------------------------------------------------------------------------------------------- """
elif test_case == 1:

    # selbstangelegte probleme: ['mehrotra_std', 'mehrotra', 'kanzow1',]
    problem = 'mehrotra_std'
    use_sparse = True

    if  problem == 'mehrotra_std':
        c = np.array([5, 3, 3, 6, 0, 0, 0])
        A_eq = np.array([[-6, 1, 2, 4, 1, 0, 0],
                        [3, -2, -1, -5, 0, 1, 0],
                        [-2, 1, 0, 2, 0, 0, 1]])
        b_eq = np.array([14, -25, 14])
        A_ineq = None
        b_ineq = None
        bounds = []

    elif problem == 'mehrotra':
        c = np.array([5, 3, 3, 6])
        A_eq = None
        b_eq = None
        A_ineq = np.array([[-6, 1, 2, 4],
                        [3, -2, -1, -5],
                        [-2, 1, 0, 2]])
        b_ineq = np.array([14, -25, 14])
        bounds = []


    if use_sparse:
        if A_eq is not None:
            A_eq = spa.csc_matrix(A_eq)
        if A_ineq is not None:
            A_ineq = spa.csc_matrix(A_ineq)


        A_std, b_std, c_std, transformations, sol_length, use_sparse = mt.lp_to_standardform(
        c=c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, verbose=verbose)
    
    
    # Anwendung von scipy.optimize.linprog auf das Ausgangsproblem
    result1 = sp.optimize.linprog(c, A_ub=A_ineq, b_ub=b_ineq, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    print(result1.x)
    print(np.dot(c, result1.x))


    # Anwendung von scipy.optimize.linprog auf die Standardform des Problems
    result2 = sp.optimize.linprog(c_std, A_eq = A_std, b_eq = b_std, method='highs')
    print(result2.x)
    print(np.dot(c_std, result2.x))

    # Zurückkonvertierte Lösung
    result2back, slack = mt.standardform_to_lp(x_std=result2.x, transformations=transformations, initial_length=sol_length, verbose=verbose)
    print(result2back)
    print(np.dot(c, result2back))


    # Anwendung von burke_xu_lp auf das Ausgangsproblem
    result3back = lp.burke_xu_lp(c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, maxiter=maxiter, acc=acc, verbose=verbose)

    if A_eq is not None:
        if use_sparse:
            print(f"A_eq hatte die Form")
            print(f"{pd.DataFrame(A_eq.toarray())}")
        else:
            print(f"A_eq hatte die Form")
            print(f"{pd.DataFrame(A_eq)}")
        print(f"b_eq hatte die Form {b_eq}.")
    if A_ineq is not None:
        if use_sparse:
            print(f"A_ineq hatte die Form")
            print(f"{pd.DataFrame(A_ineq.toarray())}")
        else:
            print(f"A_ineq hatte die Form")
            print(f"{pd.DataFrame(A_ineq)}")
        print(f"b_ineq hatte die Form {b_ineq}.")
    print(f"bounds hat die Form {bounds}.")
    print(f"Die in Standardform übeführte Matrix hatte die Form")
    print(f"{pd.DataFrame(A_std.toarray())}")
    print(f"Der Lösungsvektor lautet {result3back[0]}.")
    print(f"Der Minimale Funktionswert lautet {np.dot(c, result3back[0])} = {result3back[2]} mit Genauigkeit {acc:.0e}.")
    print(f"Es wurde {result3back[3]} mal der Nullstep verwendet, also der Prädiktor-Schritt abgelehnt.")
    print(f"Insgesamt wurden {result3back[4]} Schritte verwendet, wobei {maxiter} die maximale Anzahl der Schritte war.")
    print(f"Es wurden {result3back[5]} Sekunden benötigt.")
    
    print("----------------------------------------")

    print("Linprog auf das Ausgangsproblem ergab:")
    print(result1.x)
    print(np.dot(c, result1.x))
    print("Linprog auf das Standardproblem ergab:")
    print(result2back)
    print(np.dot(c, result2back))


    """ ---------------------------------------------------------------------------------------------------------------------- """
    """ Test verschiedener Box-Restriktionen """
    """ ---------------------------------------------------------------------------------------------------------------------- """
elif test_case == 2:

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

    A_std, b_std, c_std, transformations, sol_length, use_sparse = mt.lp_to_standardform(
        c=c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, verbose=True)

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