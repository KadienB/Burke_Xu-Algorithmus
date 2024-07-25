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
import time

data=np.load("free_for_all_qpbenchmark-main/databackup/LIPMWALK0.npz",allow_pickle=True)
data1=np.load("free_for_all_qpbenchmark-main/databackup/GNAR0.npz",allow_pickle=True)

# x = solve_qp(P, q, G, h, solver="proxqp")
# print(f"QP solution: {x = }")

def burke_xu(
    Q: Union[np.ndarray, spa.csc_matrix] = None,
    c: np.ndarray = None,
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

        Wenn A, b, G, h, lb und ub nicht übergeben werden wird LCP(c,Q) gelöst, was einem Quadratischen Programm ohne Nebenbedingungen entspricht.
    c :
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
    
    start_time = time.time()

    # Umformung des QP mit Gleichungs-Restriktionen und Box-Restriktionen zu einem QP(Q,c,A,b) mit Ungleichungsrestriktionen.
    if A == None and b != None:
        print("Wenn b angegeben wird, muss auch A angegeben werden, sonst wird b ignoriert.")
        del b
    if G == None and h != None:
        print("Wenn h angegeben wird, muss auch G angegeben werden, sonst wird h ignoriert.")
        del h
    if lb == None and ub != None:
        print("Wenn ua angegeben wird, muss auch lb angegeben werden, sonst wird ub ignoriert.")
        del ub

    if G != None:
        if h != None:
            if A == None:
                print("noch nicht implementiert")
                # A = np.block
                # b = np.block
            else:
                print("noch nicht implementiert")
                # A = np.block
                # b = np.block
        else:
            print("Wenn G angegeben wird, muss auch h angegeben werden, sonst wird G ignoriert.")
            del G

    if lb != None:
        if ub != None:
            if A == None:
                print("noch nicht implementiert")
                # A = np.block
                # b = np.block
            else:
                print("noch nicht implementiert")
                # A = np.block
                # b = np.block
        else:
            print("Wenn lb angegeben wird, muss auch ub angegeben werden, sonst wird lb ignoriert.")
            del lb

    # Umformung von QP(Q,c,A,b) zu LCP(q,M).
    if verbose:
        if np.all(np.linalg.eigvalsh(Q) >= 0):
            print("Die Matrix Q ist positiv semidefinit. Damit sollte das auch M = [Q & -A^T \\ A & 0] sein (Für A = None folgt M = Q).")
        else:
            raise ValueError("Q muss positiv semidefinit sein.")
    if A == None:
        M = Q
    else:
        M = np.block([[Q, -A.T], [A, np.zeros((A.shape[0], A.shape[0]))]])
    del Q                     # Q und A werden hiernach nicht mehr verwendet und daher gelöscht
    del A

    # Bestimmung von (x^0, y^0)

    if initvals is None:
        if b == None:
            y = c  # entspricht q in LCP(q,M)
        else:
            y = np.block([c, -b])  # entspricht q in LCP(q,M)
        x = np.zeros_like(y)
    else:
        x = initvals
        del initvals           # initvals wird hiernach nicht mehr verwendet und daher gelöscht
        if b == None:
            y = M @ x + c
        else:
            y = M @ x + np.block([c, -b])
    del c                      # c und b werden hiernach nicht mehr verwendet und daher gelöscht
    del b
    

    # Initialisierung
    k = 0
    mu = 1000
    beta = 2*len(y)
    while np.linalg.norm(m.big_phi(x, y, mu, verbose=verbose)) > beta * mu:
        beta = beta * 2
    if verbose:
        print(f"Anfangsbedingung {np.linalg.norm(m.big_phi(x, y, mu, verbose=verbose))} <= {beta * mu}")
    sigma = 0.5
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
        # x_delta = np.linalg.solve(lhs, rhs)
        y_delta = M @ x_delta
        x, y, mu, step = m.predictor_step(x, y, x_delta, y_delta, mu, alpha1, beta, acc, verbose=verbose)
        if step == 0:
            if verbose:
                print("Nullstep has been taken.")
            lhs, rhs = m.linear_equation_formulate(x, y, mu, sigma, M, 2, verbose=verbose)
            x_delta = m.linear_equation_solve(lu, piv, rhs, overwriterhs=True, verbose=verbose)
            # x_delta = np.linalg.solve(lhs, rhs)
            y_delta = M @ x_delta
        elif step == 1:
            print(f"x = {x}") 
            print(f"y = {y}")
            end_time = time.time()
            print(f"löst die KKT-Bedingungen des Quadratischen Programms, nach {k} Schritten mit Laufzeit {end_time - start_time}s.")
            break
        elif step == 2:
            lhs, rhs = m.linear_equation_formulate(x, y, mu, sigma, M, 2, verbose=verbose)
            # lu, piv = m.linear_equation_factorize(lhs, overwritelhs=True, verbose=verbose)
            # x_delta = m.linear_equation_solve(lu, piv, rhs, overwriterhs=True, verbose=verbose)
            x_delta = np.linalg.solve(lhs, rhs)
            y_delta = M @ x_delta
        x, y, mu = m.corrector_step(x, y, x_delta, y_delta, mu, alpha2, beta, sigma, verbose=verbose)
        k += 1
        if verbose:
            print(f"(x^{k},y^{k},mu_{k}) = ({x},{y},{mu})")


    # Ausgabe des Ergebnisses
        

    return

test_case = 2

    # Testbeispiel laden und printen
if test_case == 1:
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])

    Q = data['P']
    c = data['q']
    A = data['G']
    b = data['h']

    print(Q)
    print(c)
    print(A)
    print(b)

    burke_xu(Q, c, A, b, maxiter=20, verbose=True, acc=1e-8)
elif test_case == 2: # Fathi [7]
    n = 8
    n_max = 10000
    while n < n_max:
        Mn = np.identity(n)
        for i in range(1,n):
            for j in range(i):
                Mn[i,j] = 2
        print(f"{Mn}")
        M = Mn @ Mn.T
        q = np.full(n,-1)
        print(f"LCP(q,M) mit...")
        print(f"M =")
        print(M)
        print(f"q =")
        print(q)
        burke_xu(M, q, maxiter=100, verbose=False, acc=1e-8)
        print(f"gelöst für n = {n}")
        n = 2 * n