import os
from typing import Optional, Iterator, Union, Tuple
import time
import numpy as np
import scipy as sp
import scipy.sparse as spa
import sksparse.cholmod as cholmod
from memory_profiler import profile

def big_phi(
    a: np.ndarray,
    b: np.ndarray,
    mu: float,
    verbose: bool = False,
)   -> Optional[np.ndarray]:
    r"""Verwendet die Chen-Harker-Kanzow Glättungsmethode small_phi um den Vektor \Phi(x,y,mu) in |R^n wie in Kapitel 1.2 der Thesis zu erzeugen.

    Die Berechnung hat dabei Zeilenweise die Form

    a + b - sqrt((a-b)^2 + 4mu^2).

    
    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller y-Vektor in |R^n.
    mu:
        Glättungsparameter mu > 0.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    verbose = False
    if verbose:
        print("Starting big_phi calculation...")
        print(f"a = {a}")
        print(f"b = {b}")
        print(f"mu = {mu}")

    solution = (a + b) - np.sqrt(((a - b) ** 2) + (4 * (mu ** 2)))

    if verbose:
        print(f"big_phi result = {solution}")

    return solution

def nabla_big_phi(
    a: np.ndarray,
    b: np.ndarray,
    mu: float,
    arg: int,
    inv: bool = False,
    verbose: bool = False,
)   -> Optional[np.ndarray]:
    r"""Berechnung der Teilblockmatrizen von nabla(x,y)F wie in Remark 4. und 5. beschrieben unter der Verwendung von

    partial phi / partial a = 1 - (a - b) / sqrt((a - b)^2 + 4mu^2).

    partial phi / partial b = 1 + (a - b) / sqrt((a - b)^2 + 4mu^2).

    partial phi / partial mu = (-4mu) / sqrt((a - b)^2 + 4mu^2).

    Gespeichert wird jeweils in Vektorform, da es sich um vollbeschriebene Diagonalmatrizen handelt.

    
    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller y-Vektor in |R^n.
    mu:
        Glättungsparameter mu > 0.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    verbose = False
    if verbose:
        print(f"Starting nabla_big_phi calculation for argument {arg}...")
        print(f"a = {a}")
        print(f"b = {b}")
        print(f"mu = {mu}")

    bound = 1e-10
    # D = sqrt(inv(nabla_X)*nabla_S) (inv erstellen für lhs des Gleichungssystems A * X^(-1) * S * A^T = rhs)
    if arg == 0:
        sqrt_term = np.sqrt((a - b) ** 2 + (4 * mu**2))
        denominator_x_inv = np.maximum(- a + b + sqrt_term, bound)
        nabla_x_inv = np.sqrt(sqrt_term / denominator_x_inv)
        nabla_s = np.maximum(np.sqrt(1 + ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))), bound)
        diag = nabla_x_inv * nabla_s
        solution = spa.diags(diag, format='csc')

    elif arg == 1:
        if inv == False:
            solution = 1 - ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))
        else:
            sqrt_term = np.sqrt((a - b) ** 2 + (4 * mu**2))
            denominator_x_inv = np.maximum(- a + b + sqrt_term, bound)
            solution = sqrt_term / denominator_x_inv
    elif arg == 2:
        if inv == False:
            solution = 1 + ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))
        else:
            sqrt_term = np.sqrt((a - b) ** 2 + (4 * mu**2))
            solution = sqrt_term / ((a - b) + sqrt_term)
    elif arg == 3: 
        solution = (-4 * mu) / np.sqrt((a - b) ** 2 + 4 * mu ** 2)
    else: 
        raise ValueError("Argument must be 0, 1, 2 or 3.")

    if verbose:
        print(f"nabla_big_phi result = {solution}")

    return solution

def linear_equation_formulate_rhs(
    x: np.ndarray,
    a: np.ndarray,
    b: Optional[np.ndarray],
    mu: float,
    sigma: Optional[float],
    A: Union[np.ndarray, spa.csc_matrix],
    problem: int,
    use_sparse = False,
    steptype: int = 0,
    verbose: bool = False,
)   -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""Formulierung der rechten Seite des Gleichungssystem, das in jedem Prädiktor- und Korrektorschritt gelöst werden muss.
    Je nachdem ob es sich um LP, QP oder LCP handelt angepasst, für genauere Herleitung siehe Thesis Kapitel 3.

    
    Parameters
    ----------
    x: 
        in der Regel aktueller Iterationsvektor x in |R^n.
    a:
        in der Regel aktueller Iterationsvektor (s bei LP / y bei LCP) in |R^n.
    b:
        wenn vorhande, aktueller Iterationsvektor (lambda bei LP), normalerweise in |R^m.
    mu:
        Glättungsparameter mu > 0.
    A:
        Matrix (A in LP, M in LCP) von LCP(q,M) in |R^mxn.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """


    if use_sparse is False:
        if problem == 1:
            # Vorerst instabil
            D_x_inv = nabla_big_phi(x, a, mu, 1, inv=True, verbose=verbose)
            # vorerst instabil
            if steptype == 1:
                rhs = A @ np.diag(D_x_inv) @ ((big_phi(x, a, mu, verbose)) - (mu * nabla_big_phi(x, a, mu, 3, verbose)))
            elif steptype == 2:
                rhs = A @ np.diag(D_x_inv) @ ((big_phi(x, a, mu, verbose)) - (mu * sigma * nabla_big_phi(x, a, mu, 3, verbose)))
            else:
                raise ValueError("Steptype must be 1 or 2.")
            
    elif use_sparse is True:
        if problem == 1:
            D_x_inv = nabla_big_phi(x, a, mu, 1, inv=True, verbose=verbose)
            if steptype == 1:
                rhs = A.dot(spa.diags(D_x_inv).dot((big_phi(x, a, mu, verbose)) - (mu * nabla_big_phi(x, a, mu, 3, verbose))))
            elif steptype == 2:
                rhs = A.dot(spa.diags(D_x_inv)).dot((big_phi(x, a, mu, verbose)) - (mu * sigma * nabla_big_phi(x, a, mu, 3, verbose)))
            else:
                raise ValueError("Steptype must be 1 or 2.")
    
    return rhs

def cholesky_decomposition_lhs(
    x: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
    A: Union[np.ndarray, spa.csc_matrix] = None,
    problem: int = 0,
    use_sparse: bool = False,
    factor: Optional[cholmod.Factor] = None,
    regularizer: Optional[float] = 0,
    verbose: bool = False,    
)   -> Optional[Tuple[np.ndarray, np.ndarray]]:
    r"""Beschreibung

    Beschreibung

    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller y-Vektor in |R^n.
    mu:
        Glättungsparameter.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """

    min_regularizer = 1e-2

    if verbose:
        print(f"Starting cholesky_decomposition_lhs...")


        """ Wenn es sich bei A um eine dense Matrix handelt """
    if use_sparse is False:
        pass


        """ Wenn es sich bei A um eine sparse Matrix handelt """
    elif use_sparse is True:
        if factor is None:
            if verbose:
                print("Es war noch kein Factor vorhanden.")
                start_time=time.time()

            try: 
                factor = cholmod.cholesky_AAt(A, beta=0)
            except:
                while regularizer <= min_regularizer:
                    try:
                        # Versuche die Berechnung mit dem aktuellen Regularizer
                        factor = cholmod.cholesky_AAt(A, beta=regularizer)
                        if verbose:
                            print(f"Erfolgreich mit Regularizer = {regularizer}")
                        break  # Verlasse die Schleife, wenn erfolgreich
                    except Exception as e:
                        if verbose:
                            print(f"Fehler bei der Berechnung mit beta={regularizer}: {e}")
                        # Verringere den Regularizer um den Faktor 1/10
                        regularizer *= 10
                        if verbose:
                            print(f"Versuche es mit regularizer = {regularizer}")

            if verbose:
                end_time = time.time()
                print(f"Elapsed time: {end_time - start_time}")

        elif factor is not None:

            D = nabla_big_phi(x, a, mu, 0, False, verbose=verbose)

            if verbose:
                start_time=time.time()

            try:
                factor.cholesky_AAt_inplace(A.dot(D), beta=0)
            except:
                while regularizer <= min_regularizer:
                    try:
                        # Versuche die Berechnung mit dem aktuellen Regularizer
                        factor.cholesky_AAt_inplace(A.dot(D), beta=regularizer)
                        if verbose:
                            print(f"Erfolgreich mit Regularizer = {regularizer}")
                        break  # Verlasse die Schleife, wenn erfolgreich
                    except Exception as e:
                        if verbose:
                            print(f"Fehler bei der Berechnung mit beta={regularizer}: {e}")
                        # Verringere den Regularizer um den Faktor 1/10
                        regularizer *= 10
                        if verbose:
                            print(f"Versuche es mit regularizer = {regularizer}")

            if verbose:
                end_time = time.time()
                print(f"Elapsed time: {end_time - start_time}")

        return factor

    return None

def predictor_step(
    x: np.ndarray,
    a: np.ndarray,
    b: Optional[np.ndarray],
    delta_x: np.ndarray,
    delta_a: np.ndarray,
    delta_b: Optional[np.ndarray],
    mu: float,
    alpha_1: float,
    beta: float,
    acc: float,
    verbose: bool = False,    
)   -> Optional[np.ndarray]:
    r"""Beschreibung

    Beschreibung

    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller y-Vektor in |R^n.
    mu:
        Glättungsparameter.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    
    if verbose:
        print(f"Starting predictor_step calculation...")

    # if np.linalg.norm(np.minimum(x + delta_x,y + delta_y)) < acc:
    if np.linalg.norm(big_phi(x + delta_x, a + delta_a, 0)) < acc:
        step = 1
        x = x + delta_x
        a = a + delta_a
        if b is not None:
            b = b + delta_b
    elif np.linalg.norm(big_phi(x + delta_x, a + delta_a, mu, verbose=verbose)) > beta * mu:
        step = 0
        if verbose:
            print(f"Prädiktor-Schritt abgelehnt, da {np.linalg.norm(big_phi(x + delta_x, a + delta_a, mu, verbose=verbose))} > {beta * mu}")
            print(f"War der vorherige Wert in der Umgebung? {np.linalg.norm(big_phi(x , a, mu, verbose=verbose))} <= {beta * mu}")
            # print(f"Wie wäre die entgegengesetzte Richtung x? {np.linalg.norm(big_phi(x - delta_x, a + delta_a, mu, verbose=verbose))}")
            # print(f"Wie wäre die entgegengesetzte Richtung s? {np.linalg.norm(big_phi(x + delta_x, a - delta_a, mu, verbose=verbose))}")
            # print(f"Wie wäre die entgegengesetzte Richtung xs ? {np.linalg.norm(big_phi(x - delta_x, a - delta_a, mu, verbose=verbose))}")


    elif np.linalg.norm(big_phi(x + delta_x, a + delta_a, mu, verbose=verbose)) <= beta * mu:
        step = 2
        s = 1
        while np.linalg.norm(big_phi(x + delta_x, a + delta_a, (alpha_1 ** s) * mu, verbose=verbose)) <= (alpha_1 ** s) * beta * mu:
            if verbose:
                print(f"erhöhe s um 1 auf {s}, {alpha_1 ** s} noch zu groß")
            s += 1
        if verbose:
            print(f"{np.linalg.norm(big_phi(x + delta_x, a + delta_a, (alpha_1 ** (s-1)) * mu, verbose=verbose))} <= {(alpha_1 ** (s-1)) * beta * mu}")
            print(f"{np.linalg.norm(big_phi(x + delta_x, a + delta_a, (alpha_1 ** (s)) * mu, verbose=verbose))} > {(alpha_1 ** s) * beta * mu}") 
            print(f"daher eta_k = {alpha_1 ** (s-1)}")
        x = x + delta_x
        a = a + delta_a
        mu = (alpha_1 ** (s-1)) * mu
        if b is not None:
            b = b + delta_b
    
    if b is not None: 
        return x, a, b, mu, step

    return x, a, mu, step

def corrector_step(
    x: np.ndarray,
    a: np.ndarray,
    b: Optional[np.ndarray],
    delta_x: np.ndarray,
    delta_a: np.ndarray,
    delta_b: Optional[np.ndarray],
    mu: float,
    alpha_2: float,
    beta: float,
    sigma: float,
    verbose: bool = False,    
)   -> Optional[np.ndarray]:
    r"""Beschreibung

    Beschreibung

    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller y-Vektor in |R^n.
    mu:
        Glättungsparameter.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    
    if verbose:
        print(f"Starting corrector_step calculation...")

    t = 0
    while np.linalg.norm(big_phi(x + ((alpha_2 ** t) * delta_x), a + ((alpha_2 ** t) * delta_a), (1 - (sigma * (alpha_2 ** t))) * mu, verbose=verbose)) > (1 - (sigma * (alpha_2 ** t))) * beta * mu:
        t += 1
        if t == 80000:
            raise ValueError("Corrector-Step kann nicht durchgeführt werden. Run-Time-Error 't' > 80000.")
    if verbose:
        print(f"Es gilt t = {t} und damit Faktor = {(1 - (sigma * (alpha_2 ** t)))}")

    x = x + ((alpha_2 ** t) * delta_x)
    a = a + ((alpha_2 ** t) * delta_a)
    mu = (1 - (sigma * (alpha_2 ** t))) * mu
    if b is not None:
        b = b + ((alpha_2 ** t) * delta_b)
        return x, a, b, mu

    return x, a, mu

def lp_to_standardform(
    c: np.ndarray,
    A_eq: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b_eq: Optional[np.ndarray] = None,
    A_ineq: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b_ineq: Optional[np.ndarray] = None,
    bounds: Optional[np.ndarray] = None,
    verbose: bool = False,
)   -> Optional[np.ndarray]:
    r"""Beschreibung

    Die Methode wandelt ein beliebiges Lineares Programm (in Ausgangsform) in ein Lineares Programm in Standardform um.
    
    Die Eingabeparameter erfüllen dabei die Form:

    min f(x) = c^T * x


    u.d.N.

    A_eq * x = b_eq


    A_ineq * x =< b_ineq


    bounds = (lb, ub)

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
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    
    if verbose:
        print(f"Starting lp_to_standardform calculation...")

    # Bestimmen, ob die Ausgabe als sparse matrix (csc) formatiert sein soll
    if isinstance(A_eq, spa.csr_matrix):
        A_eq = A_eq.tocsc()
    if isinstance(A_ineq, spa.csr_matrix):
        A_ineq = A_ineq.tocsc()

    if isinstance(A_eq, spa.csc_matrix) == isinstance(A_ineq, spa.csc_matrix):
        use_sparse = isinstance(A_eq, spa.csc_matrix)
    elif A_eq is None and A_ineq is not None:
        use_sparse = isinstance(A_ineq, spa.csc_matrix)
    elif A_ineq is None and A_eq is not None:
        use_sparse = isinstance(A_eq, spa.csc_matrix)
    elif A_eq is None and A_ineq is None:
        use_sparse = True
    else:
        raise ValueError("One Input Matrix is 'csc', while the other one is an np.array.")
    
    # Sicherstellung des richtigen Datentyps
    if use_sparse is False:
        A_eq = A_eq.astype(np.float64)
        A_ineq = A_ineq.astype(np.float64)
    b_eq = b_eq.astype(np.float64)
    c = c.astype(np.float64)

    # Anzahl der Variablen
    initial_length = len(c)


    """ Hinzufügen der Equality Constraints """

    # Initialisierung von A_std, b_std und c_std
    if A_eq is not None:
        A_std = A_eq
        b_std = b_eq
    else:
        if use_sparse is False:
            A_std = np.empty((0, c.size))
            b_std = np.empty(0)
        elif use_sparse is True:
            A_std = spa.csc_matrix((0, c.size)) 
            b_std = np.empty(0)
    c_std = c


    """ Hinzufügen der Inequality Constraints """

    if A_ineq is not None:
        c_std = np.hstack((c_std, np.zeros(A_ineq.shape[0])))

        # Blockmatrix erstellen
        if use_sparse is False:
            A_eq_block = np.hstack([A_std, np.zeros((A_std.shape[0], A_ineq.shape[0]))])
            A_ineq_block = np.hstack([A_ineq, np.eye(A_ineq.shape[0], A_ineq.shape[0])])
            A_std = np.vstack([A_eq_block, A_ineq_block])
            b_std = np.hstack([b_std, b_ineq])

        # Sparse Blockmatrix erstellen
        elif use_sparse is True:
            A_eq_block = spa.hstack([A_std, spa.csc_matrix((A_std.shape[0], A_ineq.shape[0]))])
            A_ineq_block = spa.hstack([A_ineq, spa.eye(A_ineq.shape[0], A_ineq.shape[0])])
            A_std = spa.vstack([A_eq_block, A_ineq_block]).tocsc()
            b_std = np.hstack([b_std, b_ineq])


    """ Hinzufügen der Box Constraints """
    # Dictionary für Transformationen
    transformations = {}
    if bounds is not None and len(bounds) > 0:

        
        # Iteration über das bounds 2-Dim np.ndarray mit Fallunterscheidung
        for i in range(initial_length):
            lb, ub = bounds[i]

            # x <= np.inf
            if ub is None or np.isinf(ub):

                # Fall 1: 0 <= x <= np.inf
                if lb == 0:
                    pass

                # Fall 2: -np.inf <= x <= np.inf
                elif lb is None or np.isinf(lb):

                    # i-te Spalte negieren und rechts anfügen
                    if use_sparse is False:
                        A_std = np.hstack([A_std, -A_std[:, i][:, np.newaxis]])
                    elif use_sparse is True:
                        A_std = spa.hstack([A_std, -A_std.getcol(i)]).tocsc()
                    c_std = np.append(c_std, -c_std[i])
                    
                    # transformation eintragen
                    transformations[i] = (2, A_std.shape[1]-1)

                # Fall 3: lb <= x <= np.inf
                elif lb is not None:
                    
                    # substutiere x' = x - lb, was dazu führt dass man die i-te Spalte mit lb multipliziert auf b_std addieren muss
                    if use_sparse is False:
                        b_std += A_std[:, i] * -lb
                    elif use_sparse is True:
                        b_std += A_std[:, i].toarray().ravel() * -lb

                    # transformation eintragen
                    transformations[i] = (3, lb)
            
            # x <= ub
            elif ub is not None:

                # Fall 4: 0 <= x <= ub
                if lb == 0:
                    
                    # neue Zeile für x <= ub hinzufügen
                    if use_sparse is False:
                        new_row = np.zeros((1, A_std.shape[1]))
                        new_row[0,i] = 1
                        A_std = np.vstack([A_std, new_row])
                        new_column = np.zeros((A_std.shape[0], 1))
                        new_column[A_std.shape[0] - 1] = 1
                        A_std = np.hstack([A_std, new_column])
                        b_std = np.append(b_std, ub)
                    if use_sparse is True:
                        new_row = np.zeros((1, A_std.shape[1]))
                        new_row[0,i] = 1
                        A_std = spa.vstack([A_std, spa.csc_matrix(new_row)]).tocsc()
                        new_column = np.zeros((A_std.shape[0], 1))
                        new_column[A_std.shape[0] - 1] = 1
                        A_std = spa.hstack([A_std, spa.csc_matrix(new_column)]).tocsc()
                        b_std = np.append(b_std, ub)
                    c_std = np.append(c_std, 0)

                # Fall 5: -np.inf <= x <= ub    
                elif lb is None or np.isinf(lb):
                    
                    # Vorzeichenwechsel der Spalte i
                    if use_sparse is False:
                        A_std[:, i] = -A_std[:, i]
                    elif use_sparse is True:
                        A_std[:, i] = -A_std[:, i].toarray()
                    c_std[i] = -c_std[i]

                    # damit behandeln wir den Fall -ub <= -x <= np.inf, also wie bei Fall 3
                    lb = -ub

                    # substutiere x' = x - lb, was dazu führt dass man die i-te Spalte mit lb multipliziert auf b_std addieren muss
                    if use_sparse is False:
                        b_std += A_std[:, i] * -lb
                    elif use_sparse is True:
                        b_std += A_std[:, i].toarray().ravel() * -lb

                    # transformation eintragen
                    transformations[i] = (5, lb)

                # Fall 6: lb <= x <= ub
                elif lb is not None:

                    # substutiere x' = x - lb, was dazu führt dass man die i-te Spalte mit lb multipliziert auf b_std addieren muss
                    if use_sparse is False:
                        b_std += A_std[:, i] * -lb
                    elif use_sparse is True:
                        b_std += A_std[:, i].toarray().ravel() * -lb
                    
                    # neue Zeile für x - lb <= ub - lb, also x <= ub hinzufügen
                    if use_sparse is False:
                        new_row = np.zeros((1, A_std.shape[1]))
                        new_row[0,i] = 1
                        A_std = np.vstack([A_std, new_row])
                        new_column = np.zeros((A_std.shape[0], 1))
                        new_column[A_std.shape[0] - 1] = 1
                        A_std = np.hstack([A_std, new_column])
                        b_std = np.append(b_std, ub-lb)
                    if use_sparse is True:
                        new_row = np.zeros((1, A_std.shape[1]))
                        new_row[0,i] = 1
                        A_std = spa.vstack([A_std, spa.csc_matrix(new_row)]).tocsc()
                        new_column = np.zeros((A_std.shape[0], 1))
                        new_column[A_std.shape[0] - 1] = 1
                        A_std = spa.hstack([A_std, spa.csc_matrix(new_column)]).tocsc()
                        b_std = np.append(b_std, ub-lb)
                    c_std = np.append(c_std, 0)

                    # transformation eintragen
                    transformations[i] = (6, lb)


    return A_std, b_std, c_std, transformations, initial_length, use_sparse

def standardform_to_lp(
    x_std: np.ndarray,
    transformations: dict,
    initial_length: int,
    verbose: bool = False,
)   -> Optional[np.ndarray]:
    r"""Beschreibung

    Die Methode wandelt den Lösungsvektor des Linearen Programms in Standardform in den Lösungsvektor des Linearen Programms in Ausgangsform um.

    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller y-Vektor in |R^n.
    mu:
        Glättungsparameter.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    
    if verbose:
        print(f"Starting standardform_to_lp...")

    # transformations dictionary durchgehen und nach Fallunterscheidung Umwandlungen rückgängig machen
    for key in transformations.keys():

        # Fall 2: Wir ziehen x_i_- von x_i_+ ab.
        if transformations[key][0] == 2:
            x_std[key] -= x_std[transformations[key][1]]

        # Fall 3: Wir addieren lb auf x', da x' = x - lb gilt.
        elif transformations[key][0] == 3:
            x_std[key] += transformations[key][1]

        # Fall 5: Wir addieren lb auf x', da x' = - x - lb gilt. Dann erhalten wir durch 
        elif transformations[key][0] == 5:
            x_std[key] += transformations[key][1]
            x_std[key] = -x_std[key]

        # Fall 6:
        elif transformations[key][0] == 6:
            x_std[key] += transformations[key][1]

    slack = x_std[initial_length:]
    x_std = x_std[:initial_length]

    return x_std, slack

# def presolve_lp(
#     A_std: Union[np.ndarray, spa.csc_matrix],
#     b_std: np.ndarray,
#     c_std: np.ndarray,
#     verbose: bool = False,    
# )   -> Optional[np.ndarray]:
#     r"""Beschreibung

#     Beschreibung

#     Parameters
#     ----------
#     a: 
#         in der Regel aktueller x-Vektor in |R^n.
#     b:
#         in der Regel aktueller y-Vektor in |R^n.
#     mu:
#         Glättungsparameter.
#     arg:
#         Integer, der aussagt nach welchem Argument abgeleitet wird.
#     verbose: bool
#         Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
#     """
    
#     if verbose:
#         print(f"Starting presolve_lp calculation...")

#     return A_std, b_std, c_std

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
"""                                  Ab hier die Methoden die für die Implementierung des Algorithmus für LCPs benutzt wurden und nicht mehr verwendet werden.                                          """
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #

def linear_equation_formulate_lhs(
    x: np.ndarray,
    a: np.ndarray,
    b: Optional[np.ndarray],
    mu: float,
    A: Union[np.ndarray, spa.csc_matrix],
    problem: int,
    verbose: bool = False,    
)   -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""Formulierung der linken Seite des Gleichungssystem, das in jedem Prädiktor- und Korrektorschritt gelöst werden muss.
    Je nachdem ob es sich um LP, QP oder LCP handelt angepasst, für genauere Herleitung siehe Thesis Kapitel 3.

    
    Parameters
    ----------
    x: 
        in der Regel aktueller Iterationsvektor x in |R^n.
    a:
        in der Regel aktueller Iterationsvektor (s bei LP / y bei LCP) in |R^n.
    b:
        wenn vorhande, aktueller Iterationsvektor (lambda bei LP), normalerweise in |R^m.
    mu:
        Glättungsparameter mu > 0.
    A:
        Matrix (A in LP, M in LCP) von LCP(q,M) in |R^mxn.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """

    if problem == 1:
        # Vorerst instabil
        D_s = nabla_big_phi(x, a, mu, 2, verbose=verbose)
        # vorerst instabil 
        D_x_inv = nabla_big_phi(x, a, mu, 1, inv=True, verbose=verbose)
        lhs = A @ np.diag(D_x_inv) @ np.diag(D_s) @ A.T


    if verbose:
        print(f"Die Koeffizientenmatrix hat die Form: lhs =")
        print(f"{lhs}")
    
    return lhs

def linear_equation_formulate(
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
    sigma: float,
    M: Union[np.ndarray, spa.csc_matrix],
    arg: int,
    verbose: bool = False,    
)   -> Optional[Tuple[np.ndarray, np.ndarray]]:
    r"""Formulierung des zu lösenden LGS unter Berücksichtigung von Remarks 4 und 5 und unter Verwendung des Ansatzes des Schurkomplements für lineare Gleichungssysteme mit Blockmatrizen.

    Es genügt also in jedem Schritt das LGS

    (nabla_x(PHI) + nabla_y(PHI) @ M) @ delta_x = -PHI + mu * (sigma) * nabla_mu(PHI)

    zu lösen.

    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^(n+m).
    b:
        in der Regel aktueller y-Vektor in |R^(n+m).
    mu:
        Glättungsparameter.
    sigma:
        Zentrierungparameter.
    M:
        Matrix M von LCP(q,M) in |R^((n+m)x(n+m)).
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """

    if verbose:
        print(f"Starting linear_equation_formulate calculation for argument {arg}...")

    if verbose:
        if np.all(np.linalg.eigvalsh(np.diag(nabla_big_phi(x, y, mu, 1, verbose))) > 0):
            print("Die Matrix D_x ist positiv definit.")
        else:
            print("Die Matrix D_x ist nicht positiv definit.")
            print("x sah wie folgt aus:")
            print(f"{x}")
            print("y sah wie folgt aus:")
            print(f"{y}")
            print(f"und mu = {mu}")
            for s in range (0, len(nabla_big_phi(x, y, mu, 1, verbose))):
                if nabla_big_phi(x, y, mu, 1, verbose)[s] == 0:
                    print(f"{nabla_big_phi(x, y, mu, 1, verbose)[s]} = 1 - ({x[s]} - {y[s]}) / {np.sqrt((x[s]-y[s])**2 + 4*mu**2)} in Zeile {s}")
    if verbose:
        if np.all(np.linalg.eigvalsh(np.diag(nabla_big_phi(x, y, mu, 2, verbose))) > 0):
            print("Die Matrix D_y ist positiv definit.")
        else:
            print("Die Matrix D_y ist nicht positiv definit.")
            print(f"{np.diag(nabla_big_phi(x, y, mu, 2, verbose))}")

    lhs = np.diag(nabla_big_phi(x, y, mu, 1, verbose)) + np.diag(nabla_big_phi(x, y, mu, 2, verbose)) @ M
    if arg == 1:
        rhs = -1 * big_phi(x, y, mu, verbose) + mu * nabla_big_phi(x, y, mu, 3, verbose)
    elif arg == 2:
        rhs = -1 * big_phi(x, y, mu, verbose) + (mu * sigma * nabla_big_phi(x, y, mu, 3, verbose))
    else: 
        raise ValueError("Argument must be 1 or 2.")

    if verbose:
        print("Das LGS hat die Form:")
        print(f"{lhs}(delta_x) = {rhs}^T")

    return lhs, rhs



def linear_equation_factorize(
    lhs: Union[np.ndarray, spa.csc_matrix],
    overwritelhs: bool = False,
    verbose: bool = False,    
)   -> Optional[Tuple[np.ndarray, np.ndarray]]:
    r"""Beschreibung

    Beschreibung

    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller y-Vektor in |R^n.
    mu:
        Glättungsparameter.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    verbose = False
    if verbose:
        print(f"Starting linear_equation_factorize calculation...")

    # lu, piv = lu_factor(lhs, overwrite_a=overwritelhs)

    if verbose:
        print("Die Faktorisierung hat die Form:")
        print(f"{lu}")

    return # lu, piv

def linear_equation_solve(
    lu: Union[np.ndarray, spa.csc_matrix],
    piv: np.ndarray,
    rhs: np.ndarray,
    overwriterhs: bool = False,
    verbose: bool = False,    
)   -> Optional[np.ndarray]:
    r"""Beschreibung

    Beschreibung

    Parameters
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller y-Vektor in |R^n.
    mu:
        Glättungsparameter.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    verbose = False
    if verbose:
        print(f"Starting linear_equation_solve calculation...")

    # x = lu_solve((lu, piv), rhs, overwrite_b=overwriterhs)

    if verbose:
        print("Die Lösung x_delta lautet:")
        print(x)

    return # x

