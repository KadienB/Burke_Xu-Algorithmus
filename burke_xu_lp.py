import os
import numpy as np
import scipy as sp
import methods as mt
import time
from typing import Optional, Iterator, Union

def burke_xu_lp(
    c: np.ndarray = None,
    A: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    G: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    maxiter: Optional[int] = 10000,
    acc: Optional[float] = 1e-8,
    scaling: Optional[int] = 0,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Ein Algorithmus zur Lösung Linearer Programme nach dem Algorithmus von Burke und Xu für monotone LCP.


        Die Linearen Programme sind dabei von der Form:

        min f(x) = c^Tx 

        u.d.N.

        Ax = b 

        Gx <= h

        lb <= x <= ub


        Parameters
        ----------
        c :
            Vektor in |R^n.
        A :
            Matrix für Gleichungs-Restriktionen in |R^(mxn).
            Kann als sparse Matrix angegeben werden.
        b :
            Vektor für Gleichungs-Restriktionen in |R^m.
        G :
            Matrix für Ungleichungs-Restriktionen in |R^(sxn).
            Kann als sparse-Matrix angegeben werden.
        h :
            Vektor für Ungleichungs-Restriktionen in |R^s.
        lb :
            Untere Schranke für Box-Restriktionen in |R^n. Kann auch ``-np.inf`` sein.
            Wird nichts angegeben, wird ``0`` als untere Schranke verwendet.
        ub :
            Obere Schranke für Box-Restriktionen in |R^n. Kann auch ``+np.inf`` sein.
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


    """ Umformung des Linearen Programms in ein Lineares Programm in Normalform """

    # Ungleichungsrestriktionen zu Gleichungsrestriktionen

    # Box-Restriktionen zu Gleichungsrestriktionen


    """ Initialisierung des Algorithmus """

    # Eventuelle Erstellung der Skalierungsmatrizen
    S0 = None
    S1 = None
    S2 = None

    # Startwerte der Iterationsvariablen
    x = np.linalg.inv(A) @ b
    l = np.zeros(A.shape[1])
    s = c
    if verbose:
        print(f"Ax - b = {A @ x - b}")
        print(f"A^T*lambda + s - c = {A.T @ l + s - c}")

    mu = 1000000
    if np.linalg.norm(mt.big_phi(x, s, 0)) < acc:
        print(f"initval was solution")
        maxiter = 0

    # Festsetzung der Stellschrauben
    beta = 1000
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
        delta_x = (-1 * np.diag(mt.nabla_big_phi(x, s, mu, 2, inv=True, verbose=verbose)) @ delta_s) + (mt.big_phi(x, s, mu, verbose=verbose)) + (-1 * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose))
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
            delta_x = (-1 * np.diag(mt.nabla_big_phi(x, s, mu, 2, inv=True, verbose=verbose)) @ delta_s) + (mt.big_phi(x, s, mu, verbose)) + (- 1 * mu * sigma * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose))
        elif step == 1:
            print(f"-----------------------------------------------------------")
            print(f"x^{k + 1} = {x}")
            print(f"s^{k + 1} = {y}")
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
            delta_x = (-1 * np.diag(mt.nabla_big_phi(x, s, mu, 2, inv=True, verbose=verbose)) @ delta_s) + (mt.big_phi(x, s, mu, verbose)) + (- 1 * mu * sigma * mt.nabla_big_phi(x, s, mu, 3, verbose))
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

# Koeffizienten der Zielfunktion
c = np.array([-1, -2])

# Koeffizientenmatrix der Gleichungsnebenbedingungen (linke Seite)
A_eq = np.array([[1, 1],
                 [1, 2]])

# Rechte Seite der Gleichungsnebenbedingungen
b_eq = np.array([2, 3])

# Lösen des linearen Programms
result = sp.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

my_result = burke_xu_lp(c = c, A = A_eq, b = b_eq, maxiter=25, acc=1e-8, verbose=True)
print(f"my_result = {my_result}")

# Ausgabe der Ergebnisse
print("Lösungsstatus:", result.message)
print("Optimale Lösung x:", result.x)
print("Optimaler Zielfunktionswert:", result.fun)