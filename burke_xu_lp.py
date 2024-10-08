from typing import Optional, Tuple, Union
import time
import numpy as np
import scipy as sp
import scipy.sparse as spa
import methods as mt
from scipy.optimize._linprog_util import (_presolve, _LPProblem, _autoscale, _unscale, _clean_inputs)

def burke_xu_lp(
    c: np.ndarray = None,
    A_eq: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    A_ineq: Optional[Union[np.ndarray, sp.sparse.csc_matrix]] = None,
    b_ineq: Optional[np.ndarray] = None,
    bounds: Optional[np.ndarray] = None,
    maxiter: Optional[int] = 10000,
    acc: Optional[float] = 1e-8,
    regularizer: Optional[float] = None,
    presolve: Optional[Tuple[bool, bool, str]] = (False, False, "some_string"),
    scaling: Optional[int] = 0,
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


        Parameter
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
            Maximale Anzahl an Iterationen.
        acc :
            Gewünschte Genauigkeit.
        regularizer :
            Vorgegebener Vorfaktor der auf das Gleichungssystem addierten Einheitsmatrix. Standardmäßig wird reguralizer = acc^2 verwendet.
        presolve:
            Einstellungen für _presolve von scipy.
        scaling :
            Integer, der aussagt welche Skalierungsmethode verwendet werden soll.
        sigma :
            Externer Parameter, standardmäßig wird sigma = 0.5 verwendet.
        alpha_1 :
            Externer Parameter, standardmäßig wird alpha_1 = 0.9 verwendet.
        alpha_2 :
            Externer Parameter, standardmäßig wird alpha_2 = 0.8 verwendet.
        verbose :
            Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """


    """ Presolving des linearen Programms (mit _presolve von scipy)"""

    c_save = c
    if presolve[0] is True:
        linprog = _LPProblem(c, A_ineq, b_ineq, A_eq, b_eq, bounds)
        if verbose:
            print(f"Starting _presolve from scipy calculation...")
        linprog = _clean_inputs(linprog)
        linprog, c0, x, revstack, complete, status, message = _presolve(linprog, presolve[1], presolve[2], acc)
        if verbose:
            print(f"c0 = {c0}")
            print(f"complete = {complete}")
            print(f"status = {status}")
            print(f"message = {message}")
        c = linprog.c
        A_eq = linprog.A_eq
        b_eq = linprog.b_eq
        A_ineq = linprog.A_ub
        b_ineq = linprog.b_ub
        bounds = linprog.bounds


    """ Lineares Programm auf Normalform bringen (mit eigener Methode) """
    A, b, c, transformations, initial_length, use_sparse = mt.lp_to_standardform(c=c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, bounds=bounds, verbose=verbose)


    """ Skalierung des linearen Programms (mit _autoscale von scipy) """

    if scaling == 1:
        if verbose:
            print(f"Starting _autoscale from scipy calculation...")
        A, b, c, x0, C, b_scale = _autoscale(A, b, c, None)
        if isinstance(A, spa.csr_matrix):
            A = A.tocsc()


    """ Initialisierung des Algorithmus """
    if regularizer is None:
        regularizer = acc*acc

    # Startwerte der Iterationsvariablen
    if use_sparse is False:
        chol, low = sp.linalg.cho_factor(A.dot(A.T))
        x = sp.linalg.cho_solve((chol, low), b)
        x = A.T.dot(x)
        l = sp.linalg.cho_solve((chol, low), A.dot(c))
        s = c - A.T.dot(l)
    elif use_sparse is True:
        factor = mt.cholesky_decomposition_lhs(A=A, use_sparse=use_sparse, regularizer=regularizer, verbose=verbose)
        x = factor(b)
        x = A.T.dot(x)
        l = factor(A.dot(c))
        s = c - A.T.dot(l)

    mu = np.sqrt(acc + np.max(np.where((x > 0) & (s > 0), x * s, 0)))
    beta = np.linalg.norm(mt.big_phi(x, s, mu, verbose=verbose)) / mu

    # Externe Variablen
    if sigma is None:
        sigma = 0.5
    if alpha_1 is None:
        alpha_1 = 0.9
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

    """ Speicher aufräumen """
    del A_eq, b_eq, A_ineq, b_ineq, bounds
    if presolve[1] is True:
        del linprog


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

        # Ausgabe der aktuellen Iterationsvariablen bei Verbose
        if verbose:
            print(f"-----------------------------------------------------------")
            print(f"x^{k} = {x}")
            print(f"lambda^{k} = {l}")
            print(f"s^{k} = {s}")
            print(f"mu_{k} = {mu}")
            print(f"-----------------------------------------------------------")


        """ Prädiktor-Schritt """

        # Berechnung der mehrfach verwendeten Diagonalmatrizen und Vektoren. Die Diagonalmatrizen werden als Vektoren gespeichert, damit Speicher gespart und später eine komponentenweise
        # Multiplikation, statt einer Matrix-Vektor-Multiplikaton durchgeführt wird.
        D_x_inv = mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose)
        D_s = mt.nabla_big_phi(x, s, mu, 2, False, verbose=verbose)
        r_3 = mt.big_phi(x, s, mu, verbose) - (mu * mt.nabla_big_phi(x, s, mu, 3, verbose))
        rhs = A.dot(D_x_inv * r_3)
        # Lösung der Gleichungssysteme. Effizient implementiert nur für dünn besetzte Matrizen, use_sparse = False wurde nur zu Testzwecken implementiert.
        if use_sparse is False:
            chol, low = sp.linalg.cho_factor(A.dot(np.diag(D_x_inv).dot(np.diag(D_s))).dot(A.T))
            delta_l = sp.linalg.cho_solve((chol, low), rhs)
        else:
            factor = mt.cholesky_decomposition_lhs(x, s, mu, A, use_sparse, factor, regularizer=regularizer, verbose=verbose)
            delta_l = factor(rhs)
        delta_s = -A.T.dot(delta_l)
        delta_x = - D_x_inv * (D_s * delta_s + r_3)
        if verbose:
            print(f"im Prädiktorschritt hat delta_x den Wert")
            print(f"{delta_x}")
            print(f"big_phi hat die Werte = {mt.big_phi(x + delta_x, s + delta_s, mu, verbose=verbose)}")
        x, s, l, mu, step = mt.predictor_step(x, s, l, delta_x, delta_s, delta_l, mu, alpha_1, beta, acc, verbose=verbose)


        """ Korrektor-Schritt """
        if step == 0:
            # Wiederverwendung obiger Berechnungen bei Nullstep
            if verbose:
                print("Nullstep has been taken.")
            nullstep += 1
            r_3 = mt.big_phi(x, s, mu, verbose) - (mu * sigma * mt.nabla_big_phi(x, s, mu, 3, verbose))
            rhs = A.dot(D_x_inv * r_3)
            if use_sparse is False:
                delta_l = sp.linalg.cho_solve((chol, low), rhs)
            else:
                delta_l = factor(rhs)
            delta_s = -A.T.dot(delta_l)
            delta_x = -D_x_inv * (D_s * delta_s + r_3)
            if verbose:
                print(f"im Nullstepkorrektorschritt hat delta_x den Wert")
                print(f"{delta_x}")
                print(f"big_phi hat die Werte = {mt.big_phi(x + delta_x, s + delta_s, mu, verbose=verbose)}")
        elif step == 1:
            # Aktueller Vektor löst das LP
            maxiter = k + 1
            break
        elif step == 2:
            # Normaler Korrektor-Schritt, ähnlich implementiert wie beim Prädiktor-Schritt
            D_x_inv = mt.nabla_big_phi(x, s, mu, 1, inv=True, verbose=verbose)
            D_s = mt.nabla_big_phi(x, s, mu, 2, False, verbose=verbose)
            r_3 = mt.big_phi(x, s, mu, verbose) - (mu * sigma * mt.nabla_big_phi(x, s, mu, 3, verbose))
            rhs = A.dot(D_x_inv * r_3)
            if use_sparse is False:
                chol, low = sp.linalg.cho_factor(A.dot(np.diag(D_x_inv).dot(np.diag(D_s))).dot(A.T))
                delta_l = sp.linalg.cho_solve((chol, low), rhs)
            else:
                factor = mt.cholesky_decomposition_lhs(x, s, mu, A, use_sparse, factor, regularizer=regularizer, verbose=verbose)
                delta_l = factor(rhs)
            delta_s = -A.T.dot(delta_l)
            delta_x = -D_x_inv * (D_s * delta_s + r_3)
            if verbose:
                print(A.T.dot(delta_l) + delta_s)
                print(A.dot(delta_x))
                print(np.diag(mt.nabla_big_phi(x, s, mu, 1, inv=False, verbose=verbose)).dot(delta_x) + (np.diag(mt.nabla_big_phi(x, s, mu, 2, verbose=verbose)).dot(delta_s)))
                print(-(mt.big_phi(x, s, mu, verbose=verbose)) + (sigma * mu * mt.nabla_big_phi(x, s, mu, 3, verbose=verbose)))
        x, s, l, mu = mt.corrector_step(x, s, l, delta_x, delta_s, delta_l, mu, alpha_2, beta, sigma, verbose=verbose)


    """ Ausgabe des Ergebnisses """

    if verbose:
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

    phi = np.linalg.norm(mt.big_phi(x, s, mu, verbose=verbose), ord=np.inf)


    """ Überprüfung und Rücktransformation des Ergebnisses """

    # Überprüfung der KKT-Bedingungen und Berechnung des minimalen Wertes
    if verbose:
        print("||A^T * lambda + s - c|| =")
        print(np.linalg.norm(A.T.dot(l) + s - c))
        print("||A * x - b|| =")
        print(np.linalg.norm(A.dot(x) - b))
        print("||x^T*s|| =")
        print(np.linalg.norm(x.T @ s))
        print("haben x oder s negative Werte?")
        print(np.any(x < -acc) or np.any(s < -acc))

    # Rücktransformation
    if scaling == 1:
        x = _unscale(x, C, b_scale)

    result, slack = mt.standardform_to_lp(x, transformations, initial_length, verbose=verbose)

    if presolve[0] is True:
        for rev in reversed(revstack):
            result = rev(result)

    fun = np.dot(c_save, result)
    funoutput = (fun, mu, phi)

    return result, slack, funoutput, nullstep, maxiter, end_time - start_time