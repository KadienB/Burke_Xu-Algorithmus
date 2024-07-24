import os
from typing import Optional, Iterator, Union
import numpy as np
import qpbenchmark
import scipy.io as spio
import scipy.sparse as spa
from qpbenchmark.benchmark import main
from qpbenchmark.problem import Problem
from qpsolvers import solve_qp
import methods as m

data=np.load("free_for_all_qpbenchmark-main/databackup/LIPMWALK0.npz",allow_pickle=True)
data1=np.load("free_for_all_qpbenchmark-main/databackup/GNAR0.npz",allow_pickle=True)

# x = solve_qp(P, q, G, h, solver="proxqp")
# print(f"QP solution: {x = }")

def burke_xu(
    Q: Union[np.ndarray, spa.csc_matrix],
    c: np.ndarray,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    h: Optional[np.ndarray] = None,
    lb: Optional[np.ndarray] = None,
    ub: Optional[np.ndarray] = None,
    maxiter: Optional[int] = None,
    acc: Optional[float] = None,
    initvals: Optional[np.ndarray] = None,
    verbose: bool = False,
    **kwargs,
) -> Optional[np.ndarray]:
    r"""Der Algorithmus von Burke und Xu für monotone LCP implementiert für Quadratische Programme.

    Die QPs sind dabei von der Form:

    min f(x) = c^T + x^T Q x 

    Ax <= b 

    Gx = h

    lb <= x <= ub


    Parameters
    ----------
    Q :
        Symmetrisch positiv definite Matrix in |R^(nxn).
    C :
        Vektor in |R^n.
    A :
        Matrix für Ungleichungs-Restriktionen in |R^(mxn).
    b :
        Vektor für Ungleichungs-Restriktionen in |R^m.
    G :
        Matrix für Gleichungs-Restriktionen in |R^(sxn).
    h :
        Vektor für Gleichungs-Restriktionen in |R^s.
    lb :
        Untere Schranke für Box-Restriktionen in |R^n. Kann auch ``-np.inf`` sein.
    ub :
        Obere Schranke für Box-Restriktionen in |R^n. Kann auch ``+np.inf`` sein.
    maxiter :
        Maximum Anzahl an Iterationen.
    acc :
        Gewünschte Genauigkeit.
    initvals :
        Kandidat für einen Startvektor x^0 in |R^(n+m) und damit y^0 = Mx + q \in |R^(n+m) für einen Warm-Start des Algorithmus.
    verbose :
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
        """
    
    # Umformung des QP mit Ungleichungs-Restriktionen, Gleichungs-Restriktionen und Box-Restriktionen zu einem QP(c,Q,A,b) wie in der Thesis beschrieben.


    # to-do 


    # Initialisierung
    M = np.block([[Q, -A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
    del Q                      # Q und A werden hiernach nicht mehr verwendet und daher gelöscht
    del A

    if initvals is None:
        y = np.block([c, -b])  # entspricht q in LCP(q,M)
        x = np.zeros_like(y)
    else:
        x = initvals
        del initvals           # initvals wird hiernach nicht mehr verwendet und daher gelöscht
        y = M @ x + np.block([c, -b])
    del c                      # c und b werden hiernach nicht mehr verwendet und daher gelöscht
    del b

    k = 0
    mu = 1
    beta = 2*len(y)
    sigma = 0.1
    alpha1 = 0.9
    alpha2 = 0.9

    if verbose:
        print("Initialisiere Algorithmus...")
        print(f"M = {M} in |R^({len(M)}x{len(M[0])})")
        print(f"x^0 = {x} in |R^{len(x)}")
        print(f"y^0 = {y} in |R^{len(y)}")
        print(f"mu^0 = {mu} > 0")
        print(f"beta = {beta}")
        print(f"sigma = {sigma}")
        print(f"alpha_1 = {alpha1}")
        print(f"alpha_2 = {alpha2}")
        

    # Ausführen des Algorithmus
    for k in range(maxiter):
        lhs, rhs = m.linear_equation_formulate(x, y, mu, sigma, M, 1, verbose=verbose)
        lu, piv = m.linear_equation_factorize(lhs, overwritelhs=True, verbose=verbose)
        x_delta = m.linear_equation_solve(lu, piv, rhs, overwriterhs=True, verbose=verbose)
        y_delta = M @ x_delta
        x, y, mu, step = m.predictor_step(x, y, x_delta, y_delta, mu, alpha1, beta, acc, verbose=verbose)
        if step == 0:
            if verbose:
                print("Nullstep has been taken.")
        elif step == 1:
            if verbose:
                print(f"{x} löst die KKT-Bedingungen des Quadratischen Programms.")
            break
        elif step == 2:
            lhs, rhs = m.linear_equation_formulate(x, y, mu, sigma, M, 1, verbose=verbose)
            lu, piv = m.linear_equation_factorize(lhs, overwritelhs=True, verbose=verbose)
            x_delta = m.linear_equation_solve(lu, piv, rhs, overwriterhs=True, verbose=verbose)
            y_delta = M @ x_delta
        x, y, mu = m.corrector_step(x, y, x_delta, y_delta, mu, alpha2, beta, sigma, verbose=verbose)
        k += 1
        print(f"(x^k,y^k,mu_k) = ({x},{y},{mu})")


    # Ausgabe des Ergebnisses
        

    return


    # Testbeispiel laden und printen
lst = data1.files
for item in lst:
    print(item)
    print(data1[item])

Q = data1['P']
c = data1['q']
A = data1['A']
b = data1['b']

print(Q)
print(c)
print(A)
print(b)

burke_xu(Q, c, A, b, maxiter=100, verbose=False, acc=1e-20)