import os
import numpy as np
import scipy as sp
import scipy.sparse as spa
import methods as mt
import time
from typing import Optional, Iterator, Union

def burke_xu_lp(
    c: np.ndarray = None,
    A_eq: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    A_ineq: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    b_ineq: Optional[np.ndarray] = None,
    bounds: Optional[np.ndarray] = None,
    maxiter: Optional[int] = 10000,
    acc: Optional[float] = 1e-8,
    scaling: Optional[int] = 0,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Ein Algorithmus zur Lösung Linearer Programme nach dem Algorithmus von Burke und Xu für monotone LCP.


        Die Linearen Programme sind dabei von der Form:

        min f(x) = c^T * x

        u.d.N.

        A_eq * x = b_eq

        A_ineq * x =< b_ineq

        lb =< x =< ub


        Parameters
        ----------
        c :
            Vektor in |R^n.
        A_eq :
            Matrix für Gleichungs-Restriktionen in |R^(mxn).
            Kann als sparse Matrix angegeben werden.
        b_eq :
            Vektor für Gleichungs-Restriktionen in |R^m.
        A_ineq :
            Matrix für Ungleichungs-Restriktionen in |R^(sxn).
            Kann als sparse-Matrix angegeben werden.
        b_ineq :
            Vektor für Ungleichungs-Restriktionen in |R^s.
        Bounds :
            Matrix in |R^(nx2) für Box-Restriktionen.
            Die linke Spalte enthält lb (lower bounds), die rechte Spalte ub (upper bounds). Die Zeile entspricht dem Index von x.
                lb :
                    Untere Schranke für Box-Restriktionen in |R^n. Kann auch ``None`` sein, in dem Fall wird ``-np.inf`` als untere Schranke verwendet und x ist in diesem Index unbeschränkt nach unten.
                    Wird lb nicht angegeben, wird ``0`` als untere Schranke verwendet.
                ub :
                    Obere Schranke für Box-Restriktionen in |R^n. Kann auch ``None`` sein, in dem Fall wird ``np.inf`` als obere Schranke verwendet und x ist in diesem Index unbeschränkt nach oben.
                    Wird ub nicht angegeben, wird ``np.inf`` als obere Schranke verwendet.
        maxiter :
            Maximum Anzahl an Iterationen.
        acc :
            Gewünschte Genauigkeit.
        scaling :
            Integer, der aussagt welche Skalierungsmethode verwendet werden soll.
        initvals :
            Kandidat für einen Startvektor x^0 in |R^n um einen Warmstart durchzuführen.
        verbose :
            Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """


    """ Presolving des linearen Programms """

    """ Lineares Programm auf Normalform bringen """

    A = A_eq
    b = b_eq

    """ Initialisierung des Algorithmus """

    # Eventuelle Erstellung und Anwendung der Skalierungsmatrizen
    S0 = None
    S1 = None
    S2 = None

    # Startwerte der Iterationsvariablen
    x = np.linalg.pinv(A) @ b
    l = np.zeros(A.shape[1])
    s = c
    if verbose:
        print(f"Ax - b = {A @ x - b}")
        print(f"A^T*lambda + s - c = {A.T @ l + s - c}")

    mu = 10
    if np.linalg.norm(mt.big_phi(x, s, 0)) < acc:
        print(f"initval was solution")
        maxiter = 0

    # Externe Variablen
    beta = 100000
    while np.linalg.norm(mt.big_phi(x, s, mu, verbose=verbose)) > beta * mu:
        beta = beta * 2
    sigma = 0.5
    alpha_1 = 0.9
    alpha_2 = 0.99


    """ Ausführung des Algorithmus """

    # Initialisierung der Startzeit und des Nullstep-Counters
    k = 0
    nullstep = 0
    start = time.time()

    # Schleife zur Ausführung des Algorithmus
    for k in range(maxiter):

        # Ausgabe der aktuellen Iterationsvariablen
        print(f"-----------------------------------------------------------")
        print(f"x^{k} = {x}")
        print(f"lambda^{k} = {l}")
        print(f"s^{k} = {s}")
        print(f"mu_{k} = {mu}")
        print(f"-----------------------------------------------------------")

        # Prädiktor-Schritt
        lhs = mt.linear_equation_formulate_lhs(x, s, l, mu, A, problem=1, verbose=verbose)
        rhs = mt.linear_equation_formulate_rhs(x, s, l, mu, sigma, A, problem=1, steptype=1, verbose=verbose)
        cholesky, low = sp.linalg.cho_factor(lhs)
        delta_l = sp.linalg.cho_solve((cholesky, low), rhs)
        delta_s = -1 * A.T @ delta_l
        delta_x = (-1 * np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose))) @ (np.diag(mt.nabla_big_phi(x, s, mu, 2, verbose=verbose)) @ delta_s + (mt.big_phi(x, s, mu, verbose=verbose)) + (-1 * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
        x, l, s, mu, step = mt.predictor_step(x, s, l, delta_x, delta_s, delta_l, mu, alpha_1, beta, acc, verbose=verbose)

        # Korrektor-Schritt
        if step == 0:
            if verbose:
                print("Nullstep has been taken.")
                print(f"-----------------------------------------------------------")
                print(f"x^{k} = {x}")
                print(f"lambda^{k} = {l}")
                print(f"s^{k} = {s}")
                print(f"mu_{k} = {mu}")
                print(f"-----------------------------------------------------------")
            nullstep += 1
            rhs = mt.linear_equation_formulate_rhs(x, s, l, mu, sigma, A, problem=1, steptype=2, verbose=verbose)
            delta_l = sp.linalg.cho_solve((cholesky, low), rhs)
            delta_s = -1 * A.T @ delta_l
            delta_x = (-1 * np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose))) @ (np.diag(mt.nabla_big_phi(x, s, mu, 2, verbose=verbose)) @ delta_s + (mt.big_phi(x, s, mu, verbose=verbose)) + (-1 * sigma * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
        elif step == 1:
            print(f"-----------------------------------------------------------")
            print(f"x^{k + 1} = {x}")
            print(f"s^{k + 1} = {s}")
            print(f"mu_{k + 1} = {mu}")
            print(f"-----------------------------------------------------------")
            maxiter = k + 1
            break
        elif step == 2:
            lhs = mt.linear_equation_formulate_lhs(x, s, l, mu, A, problem=1, verbose=verbose)
            rhs = mt.linear_equation_formulate_rhs(x, s, l, mu, sigma, A, problem=1, steptype=2, verbose=verbose)
            cholesky, low = sp.linalg.cho_factor(lhs)
            delta_l = sp.linalg.cho_solve((cholesky, low), rhs)
            delta_s = -1 * A.T @ delta_l
            delta_x = (-1 * np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose))) @ (np.diag(mt.nabla_big_phi(x, s, mu, 2, verbose=verbose)) @ delta_s + (mt.big_phi(x, s, mu, verbose=verbose)) + (-1 * sigma * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
        x, l, s, mu = mt.corrector_step(x, s, l, delta_x, delta_s, delta_l, mu, alpha_2, beta, sigma, verbose=verbose)


    """ Ausgabe des Ergebnisses """

    end_time = time.time()

    # dim_qp = len(x) - dim_diff
    # qp_sol = x[:dim_qp]
    # if scaling == 0:
    #     qp_sol_y = M @ x + q
    # else:
    #     qp_sol_y = M_old @ x + q_old
    # qp_sol_smooth = np.where(np.abs(qp_sol) < acc, 0, qp_sol)
    # qp_sol_y_smooth = np.where(np.abs(qp_sol_y) < acc, 0, qp_sol_y)

    # if verbose:
    #     print(f"Es wurde Skalierungsmethode {scaling} verwendet.")
    #     print(f"Die Genauigkeit beträgt {acc}.")
    #     print(f"Die Lösung (x,y) des LCP(q,M) wie in der Thesis beschrieben lautet")
    #     print(f"x in |R^{len(x)} =")
    #     print(x)
    #     print(f"y in |R^{len(y)} =")
    #     print(qp_sol_y)

    # print(f"x in |R^{dim_qp} =")
    # print(qp_sol)
    # print(f"Der vorhergehende Vektor x löst das zugehörige Quadratische Programm, oder im Fall 'A == None' das zugehörige lineare Komplementaritätsproblem.")
    # print(f"Es wurden dafür {maxiter} Schritte durchgeführt. Es wurden {end_time - start_time} Sekunden benötigt.")
    # print(f"Es wurde {nullstep} mal der Prediktor-Schritt abgelehnt.")


    """ Überprüfung des Ergebnisses """

    return x

test_case = 2

if test_case == 1:

    # Koeffizienten der Zielfunktion
    c = np.array([-1, -2])

    # Koeffizientenmatrix der Gleichungsnebenbedingungen (linke Seite)
    A_eq = np.array([[1, 1],
                    [1, 2]])

    # Rechte Seite der Gleichungsnebenbedingungen
    b_eq = np.array([2, 3])

    # Lösen des linearen Programms
    result = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

    my_result = burke_xu_lp(c = c, A_eq = A_eq, b_eq = b_eq, maxiter=25, acc=1e-4, verbose=True)
    print(f"my_result = {my_result}")

    # Ausgabe der Ergebnisse
    print("Lösungsstatus:", result.message)
    print("Optimale Lösung x:", result.x)
    print("Optimaler Zielfunktionswert:", result.fun)

if test_case == 2:

    c = np.array([1, 2, 3])
    A_eq = np.array([[1, 2, 3]])
    b_eq = np.array([7])
    A_ineq = np.array([[2, 2, -1], [-1, 1, 0]])
    b_ineq = np.array([10, -3])
    # lb = np.array([-1, None, 0])
    # up = np.array([13, 7, 11])
    lb = np.array([-100, 0, 0])
    up = np.array([None, None, None])
    bounds = np.vstack((lb, up)).T
    lb = np.array([0, 0, 0, 0, 0])
    up = np.array([None, None, None, None, None])
    bounds1 = np.vstack((lb, up)).T

    print(f"c = {c}")
    print(f"A_eq = {A_eq}")
    print(f"b_eq = {b_eq}")
    print(f"A_ineq = {A_ineq}")
    print(f"b_ineq = {b_ineq} len(b_ineq) = {len(b_ineq)}")
    print(f"lb = {lb}")
    print(f"up = {up}")
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
    res = mt.standardform_to_lp(x_std=result2.x, transformations=transformations, initial_length=sol_length, verbose=True)
    print(res)
    print(np.dot(c_std, res))