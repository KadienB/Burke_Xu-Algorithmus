import os
from typing import Optional, Iterator, Union, Tuple
import numpy as np
import qpbenchmark
import scipy.io as spio
import scipy.sparse as spa
from scipy.linalg import lu_factor, lu_solve
from qpbenchmark.benchmark import main
from qpbenchmark.problem import Problem
from qpsolvers import solve_qp

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
        solution = 1 - ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))
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
        D_s_inv = nabla_big_phi(x, a, mu, 2, inv=True, verbose=verbose)
        # vorerst instabil 
        D_x = nabla_big_phi(x, a, mu, 1, verbose=verbose)
        lhs = A @ np.diag(D_s_inv) @ np.diag(D_x) @ A.T


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
        D_s = nabla_big_phi(x, a, mu, 2, verbose)
        D_s_inv = 1 / D_s
        # vorerst instabil
        if steptype == 1:
            rhs = A @ np.diag(D_s_inv) @ ((big_phi(x, a, mu, verbose)) + (-1 * mu * nabla_big_phi(x, a, mu, 3, verbose)))
        elif steptype == 2:
            rhs = A @ np.diag(D_s_inv) @ ((big_phi(x, a, mu, verbose)) + (- 1 * mu * sigma * nabla_big_phi(x, a, mu, 3, verbose)))
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

# ----------------------------------------------------------------------------------------------------------------------------- #
""" Ab hier die Methoden die für die Implementierung des Algorithmus für LCPs benutzt wurden und nicht mehr verwendet werden. """
# ----------------------------------------------------------------------------------------------------------------------------- #

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

    lu, piv = lu_factor(lhs, overwrite_a=overwritelhs)

    if verbose:
        print("Die Faktorisierung hat die Form:")
        print(f"{lu}")

    return lu, piv

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

    x = lu_solve((lu, piv), rhs, overwrite_b=overwriterhs)

    if verbose:
        print("Die Lösung x_delta lautet:")
        print(x)

    return x

