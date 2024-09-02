import os
from typing import Optional, Iterator, Union
import time
import numpy as np
import scipy as sp
import scipy.sparse as spa
import sksparse as skit
import methods as mt
from memory_profiler import profile


def burke_xu_lp(
    c: np.ndarray = None,
    A_eq: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    A_ineq: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    b_ineq: Optional[np.ndarray] = None,
    bounds: Optional[np.ndarray] = None,
    maxiter: Optional[int] = 10000,
    acc: Optional[float] = 1e-8,
    regularizer: Optional[float] = 0,
    scaling: Optional[int] = 0,
    initvals: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
    alpha_1: Optional[float] = None,
    alpha_2: Optional[float] = None,
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

    c_save = c
    A, b, c, transformations, initial_length, use_sparse = mt.lp_to_standardform(c=c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, verbose=verbose)

    """ Initialisierung des Algorithmus """

    # Eventuelle Erstellung und Anwendung der Skalierungsmatrizen
    S0 = None
    S1 = None
    S2 = None
    problem = 1

    # Startwerte der Iterationsvariablen
    factor = mt.cholesky_decomposition_lhs(A=A, problem=1, use_sparse=use_sparse, regularizer=regularizer, verbose=verbose)
    if use_sparse is False:
        pass
    elif use_sparse is True:
        x = factor(b)
        x = A.T.dot(x)
    # l = np.zeros(A.shape[0])
    # s = c
    l = factor(A.dot(c))
    s = c - A.T.dot(l)


    # Externe Variablen
    mu = np.sqrt(np.max(np.maximum(0, x * s)))*1.1
    if np.linalg.norm(mt.big_phi(x, s, 0)) < acc:
        print(f"initval was solution")
        maxiter = 0
    beta = np.linalg.norm(mt.big_phi(x, s, mu, verbose=verbose)) / mu
    while beta < 2 * np.sqrt(len(x)):
        beta = beta * 1.1

    if sigma is None:
        sigma = 0.5
    if alpha_1 is None:
        alpha_1 = 0.75
    if alpha_2 is None:
        alpha_2 = 0.8

    if verbose:
        print(f"mu = {mu}")
        print(f"beta = {beta}")
        print(f"sigma = {sigma}")
        print(f"alpha_1 = {alpha_1}")
        print(f"alpha_2 = {alpha_2}")

        print(f"Ax - b = {np.linalg.norm(A.dot(x) - b)}")
        print(f"A^T*lambda + s - c = {np.linalg.norm(A.T.dot(l) + s - c)}")
        print(f"{np.linalg.norm(mt.big_phi(x, s, mu))} <= {beta * mu}")
        print(f"{beta} > {2 * np.sqrt(len(x))}")
        print(f"big_phi hat nur negative komponenten ist {np.all(mt.big_phi(x, s, mu) < 0)}")
        print(f"{mt.big_phi(x, s, mu)}")


    """ Ausführung des Algorithmus """

    # Initialisierung der Startzeit und des Nullstep-Counters
    k = 0
    nullstep = 0
    start_time = time.time()

    # Schleife zur Ausführung des Algorithmus
    for k in range(maxiter):

        # Abbruch-Kriterium
        if mu < acc*acc or np.linalg.norm(mt.big_phi(x, s, 0, verbose=verbose), ord=np.inf) < acc:
            maxiter = k
            break

        # Ausgabe der aktuellen Iterationsvariablen
        print(f"-----------------------------------------------------------")
        print(f"x^{k} = {x}")
        print(f"lambda^{k} = {l}")
        print(f"s^{k} = {s}")
        print(f"mu_{k} = {mu}")
        print(f"-----------------------------------------------------------")

        # Prädiktor-Schritt
        rhs = mt.linear_equation_formulate_rhs(x, s, l, mu, sigma, A, problem, use_sparse, steptype=1, verbose=verbose)
        factor = mt.cholesky_decomposition_lhs(x, s, mu, A, problem, use_sparse, factor, regularizer=regularizer, verbose=verbose)
        # np.set_printoptions(threshold=np.inf)
        # print(A.dot(mt.nabla_big_phi(x, s, mu, 0, False, verbose=verbose)).dot(mt.nabla_big_phi(x, s, mu, 0, False, verbose=verbose).dot(A.T)).toarray())
        # print(rhs)
        # np.set_printoptions(threshold=1000)
        delta_l = factor(rhs)
        delta_s = -A.T.dot(delta_l)
        delta_x = (np.diag(-mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose))) @ ((np.diag(mt.nabla_big_phi(x, s, mu, 2, False, verbose=verbose)) @ delta_s) + (mt.big_phi(x, s, mu, verbose=verbose)) - (mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
        if verbose:
            print(f"im Prädiktorschritt hat delta_x den Wert {delta_x}")
            print(f"zusammengesetzt aus")
            print(f"X^(-1) = {mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose)}")
            print(f"S = {mt.nabla_big_phi(x, s, mu, 2, False, verbose=verbose)}")
            print(f"delta_S = {delta_s}")
            print(f"big_phi  = {mt.big_phi(x, s, mu, verbose=verbose)}")
            print(f"mu * nabla_big_phi = {mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)}")
            print(f"big_phi hat die Werte = {mt.big_phi(x + delta_x, s + delta_s, mu, verbose=verbose)}")
            print(f"Der Betrag davon lautet {np.linalg.norm(mt.big_phi(x + delta_x, s + delta_s, mu, verbose=verbose))}")
            print(f"beta und mu haben die Werte {beta} und {mu}")
            # print(A.T.dot(delta_l) + delta_s)
            # print(A.dot(delta_x))
            # print(np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=False, verbose=verbose)).dot(delta_x) + (np.diag(mt.nabla_big_phi(x, s, mu, 2, verbose=verbose)).dot(delta_s)))
            # print(-(mt.big_phi(x, s, mu, verbose=verbose)) + (mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
        x, s, l, mu, step = mt.predictor_step(x, s, l, delta_x, delta_s, delta_l, mu, alpha_1, beta, acc, verbose=verbose)

        # Korrektor-Schritt
        if step == 0:
            print("Nullstep has been taken.")
            nullstep += 1
            rhs = mt.linear_equation_formulate_rhs(x, s, l, mu, sigma, A, problem, use_sparse, steptype=2, verbose=verbose)
            delta_l = factor(rhs)
            delta_s = -A.T.dot(delta_l)
            delta_x = (-1 * np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose))) @ (np.diag(mt.nabla_big_phi(x, s, mu, 2, False, verbose=verbose)) @ delta_s + (mt.big_phi(x, s, mu, verbose=verbose)) - (sigma * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
            if verbose:
                print(f"im Nullstepkorrektorschritt hat delta_x den Wert {delta_x}")
                print(f"zusammengesetzt aus")
                print(f"X^(-1) = {mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose)}")
                print(f"S = {mt.nabla_big_phi(x, s, mu, 2, False, verbose=verbose)}")
                print(f"delta_S = {delta_s}")
                print(f"big_phi  = {mt.big_phi(x, s, mu, verbose=verbose)}")
                print(f"mu * nabla_big_phi = {mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)}")
                print(f"big_phi hat die Werte = {mt.big_phi(x + delta_x, s + delta_s, mu, verbose=verbose)}")
                print(f"Der Betrag davon lautet {np.linalg.norm(mt.big_phi(x + delta_x, s + delta_s, mu, verbose=verbose))}")
                print(f"beta und mu haben die Werte {beta} und {mu}")
                # print(A.T.dot(delta_l) + delta_s)
                # print(A.dot(delta_x))
                # print(np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=False, verbose=verbose)).dot(delta_x) + (np.diag(mt.nabla_big_phi(x, s, mu, 2, verbose=verbose)).dot(delta_s)))
                # print(-(mt.big_phi(x, s, mu, verbose=verbose)) + (sigma * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
        elif step == 1:
            maxiter = k + 1
            break
        elif step == 2:
            rhs = mt.linear_equation_formulate_rhs(x, s, l, mu, sigma, A, problem, use_sparse, steptype=2, verbose=verbose)
            factor = mt.cholesky_decomposition_lhs(x, s, mu, A, problem, use_sparse, factor, regularizer=regularizer, verbose=verbose)
            delta_l = factor(rhs)
            delta_s = -A.T.dot(delta_l)
            delta_x = (-1 * np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose))) @ (np.diag(mt.nabla_big_phi(x, s, mu, 2, verbose=verbose)) @ delta_s + (mt.big_phi(x, s, mu, verbose=verbose)) - (sigma * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
            if verbose:
                print(A.T.dot(delta_l) + delta_s)
                print(A.dot(delta_x))
                print(np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=False, verbose=verbose)).dot(delta_x) + (np.diag(mt.nabla_big_phi(x, s, mu, 2, verbose=verbose)).dot(delta_s)))
                print(-(mt.big_phi(x, s, mu, verbose=verbose)) + (sigma * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
        x, s, l, mu = mt.corrector_step(x, s, l, delta_x, delta_s, delta_l, mu, alpha_2, beta, sigma, verbose=True)


    """ Ausgabe des Ergebnisses """

    print(f"-----------------------------------------------------------")
    print(f"x^{k} = {x}")
    print(f"lambda^{k} = {l}")
    print(f"s^{k} = {s}")
    print(f"mu_{k} = {mu}")
    print(f"-----------------------------------------------------------")

    end_time = time.time()



    if verbose:

        print(A.toarray())
        print(b)
        print(f"Die Genauigkeit beträgt {acc}.")
        print(f"Es wurde {nullstep} mal der Prediktor-Schritt abgelehnt.")
        print(f"Es wurden dafür {maxiter} Schritte durchgeführt. Es wurden {end_time - start_time} Sekunden benötigt.")


    """ Überprüfung und Rücktransformation des Ergebnisses """

    # Überprüfung der KKT-Bedingungen und Berechnung des minimalen Wertes
    print("||A^T * lambda + s - c|| =")
    print(np.linalg.norm(A.T.dot(l) + s - c))
    print("||A * x - b|| =")
    print(np.linalg.norm(A.dot(x) - b))
    print("||x^T*s|| =")
    print(np.linalg.norm(x.T @ s))
    print("haben x oder s negative Werte?")
    print(np.any(x < -acc) or np.any(s < -acc))

    # Rücktransformation
    result, slack = mt.standardform_to_lp(x, transformations, initial_length, verbose=verbose)

    fun = np.dot(c_save, result)

    return result, slack, fun, nullstep, maxiter, end_time - start_time