import os
from typing import Optional, Iterator, Union, Tuple
import numpy as np
import scipy as sp
import scipy.sparse as spa
import sksparse as skit

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
    inv: Optional[bool] = False,
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

    if arg == 1:
        if inv == False:
            solution = 1 - ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))
        else:
            sqrt_term = np.sqrt((a**2) - (2 * a * b) + (b**2) + (4 * mu**2))
            solution = sqrt_term / ((- a + b) + sqrt_term)
    elif arg == 2:
        if inv == False:
            solution = 1 + ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))
        else:
            sqrt_term = np.sqrt((a**2) - (2 * a * b) + (b**2) + (4 * mu**2))
            solution = sqrt_term / ((a - b) + sqrt_term)
    elif arg == 3: 
        solution = (-4 * mu) / np.sqrt((a - b) ** 2 + 4 * mu ** 2)
    else: 
        raise ValueError("Argument must be 1, 2 or 3.")

    if verbose:
        print(f"nabla_big_phi result = {solution}")

    return solution

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

def linear_equation_formulate_rhs(
    x: np.ndarray,
    a: np.ndarray,
    b: Optional[np.ndarray],
    mu: float,
    sigma: Optional[float],
    A: Union[np.ndarray, spa.csc_matrix],
    problem: int,
    steptype: int,
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

    if problem == 1:
        # Vorerst instabil
        D_x_inv = nabla_big_phi(x, a, mu, 1, inv=True, verbose=verbose)
        # vorerst instabil
        if steptype == 1:
            rhs = A @ np.diag(D_x_inv) @ ((big_phi(x, a, mu, verbose)) + (-1 * mu * nabla_big_phi(x, a, mu, 3, verbose)))
        elif steptype == 2:
            rhs = A @ np.diag(D_x_inv) @ ((big_phi(x, a, mu, verbose)) + (- 1 * mu * sigma * nabla_big_phi(x, a, mu, 3, verbose)))
        else:
            raise ValueError("Steptype must be 1 or 2.")


    if verbose:
        print(f"Die rechte Seite lautet hat die Form: rhs =")
        print(f"{rhs}")
    
    return rhs

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
    else:
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
        return x, b, a, mu, step

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
        return x, b, a, mu

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
    num_vars = initial_length


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

    if bounds is not None:

        # Dictionary für Transformationen
        transformations = {}
        
        # Iteration über das bounds 2-Dim np.ndarray mit Fallunterscheidung
        for i in range(initial_length):
            lb, ub = bounds[i]

            # x <= np.inf
            if ub is None:

                # 0 <= x <= np.inf
                if lb == 0:
                    pass

                # -np.inf <= x <= np.inf
                elif lb is None:

                    # i-te Spalte negieren und rechts anfügen
                    if use_sparse is False:
                        A_std = np.hstack([A_std, -A_std[:, i][:, np.newaxis]])
                    elif use_sparse is True:
                        A_std = spa.hstack([A_std, -A_std.getcol(i)]).tocsc()

                # lb <= x <= np.inf
                elif lb is not None:
                    
                    # substutiere x' = x - lb, was dazu führt dass man die i-te Spalte mit lb multipliziert auf b_std addieren muss
                    if use_sparse is False:
                        b_std += A_std[:, i] * lb
                    elif use_sparse is True:
                        b_std += A_std[:, i].toarray().ravel() * lb
            
            # x <= ub
            elif ub is not None:

                # 0 <= x <= ub
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

                # -np.inf <= x <= ub    
                elif lb is None:
                    VZW_und_verschiebung = 1

                # lb <= x <= ub
                elif lb is not None:
                    ub = ub - lb
                    
                    # neue Zeile für x - lb <= ub - lb hinzufügen
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

                    # substutiere x' = x - lb, was dazu führt dass man die i-te Spalte mit lb multipliziert auf b_std addieren muss
                    if use_sparse is False:
                        b_std += A_std[:, i] * lb
                    elif use_sparse is True:
                        b_std += A_std[:, i].toarray().ravel() * lb


    return A_std, b_std, c_std, transformations, initial_length, use_sparse

def presolve_lp(
    A_std: Union[np.ndarray, spa.csc_matrix],
    b_std: np.ndarray,
    c_std: np.ndarray,
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
        print(f"Starting presolve_lp calculation...")

    return A_std, b_std, c_std

def linear_equation_factorize(
    A: Union[np.ndarray, spa.csc_matrix],
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


    if verbose:
        print("Die Faktorisierung hat die Form:")
        print(f"{lu}")

    return None

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

