import os
from typing import Optional, Iterator, Union
import numpy as np
import qpbenchmark
import scipy.io as spio
import scipy.sparse as spa
from scipy.optimize import linprog
from qpbenchmark.benchmark import main
from qpbenchmark.problem import Problem
from qpsolvers import solve_qp
import methods as m
import time

def burke_xu(
    Q: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    c: np.ndarray = None,
    A: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
    b: Optional[np.ndarray] = None,
    G: Optional[Union[np.ndarray, spa.csc_matrix]] = None,
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

    # Spezialfall: Lineares Programm
    if Q is None and c is not None:
        Q = np.zeros((len(c), len(c)))

    # Umformung des QP mit Gleichungs-Restriktionen und Box-Restriktionen zu einem QP(Q,c,A,b) mit Ungleichungsrestriktionen.
    if A is None and b is not None:
        print("Wenn b angegeben wird, muss auch A angegeben werden, sonst wird b ignoriert.")
        del b
    if G is None and h is not None:
        print("Wenn h angegeben wird, muss auch G angegeben werden, sonst wird h ignoriert.")
        del h
    if lb is None and ub is not None:
        print("Wenn ua angegeben wird, muss auch lb angegeben werden, sonst wird ub ignoriert.")
        del ub

    if G is not None:
        if h is not None:
            if A is not None:
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

    if lb is not None:
        if ub is not None:
            if A is None:
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
        if np.array_equal(Q, Q.T) is True:
            print("Die Matrix Q ist symmetrisch.")
            if np.all(np.linalg.eigvalsh(Q) >= 0):
                print("Die Matrix Q ist positiv semidefinit. Damit sollte das auch M = [Q & -A^T \\ A & 0] sein (Für A = None folgt M = Q).")
            else:
                raise ValueError("Q muss positiv semidefinit sein.")
        else: raise ValueError("Q muss symmetrisch sein.")
    if A is None:
        M = Q
        q = c
        dim_diff = 0
    else:
        M = np.block([[Q, A.T], [-A, np.zeros((A.shape[0], A.shape[0]))]])
        q = np.block([c, b])
        dim_diff = len(A)
    del Q                     # Q und A werden hiernach nicht mehr verwendet und daher gelöscht
    del A
    del c
    del b


    # Skalierungsmatrizen bestimmen
    if scaling == 1:
        S = np.diag(np.array([1/m_ii if np.absolute(m_ii) > acc else 1 for m_ii in np.diag(M)]))
        if verbose:
            print(f"Die Skalierungsmatrix S von {M} und {q} lautet:")
            print(S)
        M_old = M
        q_old = q
        M = S @ M
        q = S @ q
        del S

    elif scaling == 2:
        S = np.diag(np.array([1/np.linalg.norm(row) if np.linalg.norm(row) > acc else 1 for row in M]))      
        if verbose:
            print(f"Die Skalierungsmatrix S von {M} und {q} lautet:")
            print(S)  
        M_old = M
        q_old = q
        M = S @ M
        q = S @ q
        del S

    elif scaling == 3:
        S = np.diag(np.array([1/np.absolute(q_i) if np.absolute(q_i) > acc else 1 for q_i in q]))
        if verbose:
            print(f"Die Skalierungsmatrix S von {M} und {q} lautet:")
            print(S)
        M_old = M
        q_old = q
        M = S @ M
        q = S @ q
        del S

    elif scaling != 0:
        raise ValueError("scaling muss 0, 1, 2 oder 3 sein.")


    # Bestimmung von (x^0, y^0)
    if initvals is None:
        y = q # entspricht q in LCP(q,M)
        x = np.zeros_like(y)
    else:
        x = initvals
        del initvals           # initvals wird hiernach nicht mehr verwendet und daher gelöscht
        y = M @ x + q
    

    # Initialisierung
    mu = np.linalg.norm(q) / len(q)
    beta = 2 * np.sqrt(len(q))
    while np.linalg.norm(m.big_phi(x, y, mu, verbose=verbose)) > beta * mu:
        beta = beta * 2

    if verbose:
        print(f"Anfangsbedingung {np.linalg.norm(m.big_phi(x, y, mu, verbose=verbose))} <= {beta * mu}")

    sigma = 0.5
    alpha1 = 0.75
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


    nullstep = 0
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
            nullstep += 1
            lhs, rhs = m.linear_equation_formulate(x, y, mu, sigma, M, 2, verbose=verbose)
            x_delta = m.linear_equation_solve(lu, piv, rhs, overwriterhs=True, verbose=verbose)
            # x_delta = np.linalg.solve(lhs, rhs)
            y_delta = M @ x_delta
        elif step == 1:
            maxiter = k + 1
            break
        elif step == 2:
            lhs, rhs = m.linear_equation_formulate(x, y, mu, sigma, M, 2, verbose=verbose)
            # lu, piv = m.linear_equation_factorize(lhs, overwritelhs=True, verbose=verbose)
            # x_delta = m.linear_equation_solve(lu, piv, rhs, overwriterhs=True, verbose=verbose)
            x_delta = np.linalg.solve(lhs, rhs)
            y_delta = M @ x_delta
        x, y, mu = m.corrector_step(x, y, x_delta, y_delta, mu, alpha2, beta, sigma, verbose=verbose)

        if verbose:
            print(f"(x^{k+1},y^{k+1},mu_{k+1}) = ({x},{y},{mu})")

        del rhs
        del lhs
        del lu
        del piv
        del x_delta
        del y_delta

    # Ausgabe des Ergebnisses
    
    end_time = time.time()

    dim_qp = len(x) - dim_diff
    qp_sol = x[:dim_qp]
    if scaling == 0:
        qp_sol_y = M @ x + q
    else:
        qp_sol_y = M_old @ x + q_old
    qp_sol_smooth = np.where(np.abs(qp_sol) < acc, 0, qp_sol)

    if verbose:
        print(f"Es wurde Skalierungsmethode {scaling} verwendet.")
        print(f"Die Genauigkeit beträgt {acc}.")
        print(f"Die Lösung (x,y) des LCP(q,M) wie in der Thesis beschrieben lautet")
        print(f"x in |R^{len(x)} =")
        print(x)
        print(f"y in |R^{len(y)} =")
        print(qp_sol_y)

    print(f"x in |R^{dim_qp} =")
    print(qp_sol)
    print(f"Der vorhergehende Vektor x löst das zugehörige Quadratische Programm, oder im Fall 'A == None' das zugehörige lineare Komplementaritätsproblem.")
    print(f"Es wurden dafür {maxiter} Schritte durchgeführt. Es wurden {end_time - start_time} Sekunden benötigt.")
    print(f"Es wurde {nullstep} mal der Prediktor-Schritt abgelehnt.")

    return qp_sol, x, qp_sol_y, qp_sol_smooth, maxiter, end_time - start_time, nullstep


test_case = 1

    # Testbeispiel laden und printen
if test_case == 1: # Testbeispiele von QP-Benchmark mit verschiedenen Nebenbedingungen

    # Daten laden
    data=np.load("free_for_all_qpbenchmark-main/databackup/LIPMWALK0.npz",allow_pickle=True)
    # data=np.load("free_for_all_qpbenchmark-main/databackup/QUADCMPC1.npz",allow_pickle=True)

    # Geladenes Problem printen
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])

    # Liste von Vergleichssolvern und Problem zu QP(Q,c,A,b) umwandeln
    solvers = ['clarabel', 'cvxopt', 'daqp', 'ecos', 'highs', 'osqp', 'piqp', 'proxqp', 'qpalm', 'qpoases', 'quadprog', 'scs']
    Q = data['P']
    c = data['q']
    A = data['G']
    b = data['h']

    # Vektor für Warmstart festlegen
    init = np.ones(len(c) + len(b))

    # Algorithmus mit QPsolvern vergleichen
    x_me = burke_xu(Q, c, A, b, maxiter=10000000, verbose=False, initvals=init, acc=1e-6, scaling=0)
    print("")
    lb_test = np.zeros(len(c))
    for solver in solvers:
        x = solve_qp(P=Q, q=c, G=A, h=b, lb=lb_test, ub=None, solver=solver)
        print(x)
        print("ist die Lösung von ", solver, ".")


elif test_case == 2: # Fathi [7]
    n = 256
    n_max = 257
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
        burke_xu(M, q, maxiter=1000, verbose=False, acc=1e-6, scaling=2)
        print(f"gelöst für n = {n}")
        n = 2 * n

elif test_case == 2.5:
    # Set the random seed for reproducibility
    np.random.seed(0)

    # Define the size of the matrices and vectors
    n = 5  # You can change this value to generate different sizes

    # Generate random matrix A with elements in the range (-5, 5)
    A = np.random.uniform(-5, 5, (n, n))

    # Generate random skew-symmetric matrix B
    B = np.random.uniform(-5, 5, (n, n))
    B = B - B.T  # Make B skew-symmetric

    # Generate random vector q with elements in the range (-500, 500)
    q = np.random.uniform(-500, 500, n)

    # Generate random vector η with elements in the range (0.0, 0.3)
    eta = np.random.uniform(0.0, 0.3, n)

    # Define M using the formula M = A.T @ A + B + np.diag(η)
    M = A.T @ A + B + np.diag(eta)

    print(M)
    print(q)

    x = burke_xu(M, q, maxiter=1000, verbose=False, acc=1e-6, scaling=0)


elif test_case == 3: # Kleines Quadratisches Programm

    # Beispielwerte für Q, c, A und b als np.ndarray ohne buffer
    Q = np.ndarray(shape=(2, 2))
    Q[:] = [[1, 0], [0, 2]]  # Matrixzuweisung

    c = np.ndarray(shape=(2,))
    c[:] = [-2, -6]  # Vektorzuweisung

    A = np.ndarray(shape=(3, 2))
    A[:] = [[1, 1], [-1, 2], [2, 1]]  # Matrixzuweisung

    b = np.ndarray(shape=(3,))
    b[:] = [2, 2, 3]  # Vektorzuweisung

    print(Q)
    print(c)
    print(A)
    print(b)

    x_me = burke_xu(Q, c, A, b, maxiter=100, verbose=True, acc=1e-6, scaling=3)
    x_qp = solve_qp(P=Q, q=c, G=A, h=b, solver="proxqp")
    
    print(f"Meine Lösung lautet x = {x_me}")
    print(f"proxqp ergab x = {x_qp}")


elif test_case == 4: # Lineares Programm

    n = 5
    s = 3
    lb_test = np.zeros(n)

    # Generiere zufällige Zielfunktionskoeffizienten c
    np.random.seed(0)  # Für Reproduzierbarkeit
    c = np.random.rand(n)

    # Generiere eine zufällige Matrix A
    A = np.random.rand(s, n)

    # Generiere einen zufälligen Vektor b
    b = np.random.rand(s)

    Q = np.zeros((n,n))

    x_me = burke_xu(c=c, A=A, b=b, maxiter=100, verbose=True, acc=1e-6, scaling=0)
    x_lp = linprog(c, A_ub=A, b_ub=b, bounds=(0, None))
    
    print(f"Meine Lösung lautet x = {x_me}")
    print(f"scipy ergab x = {x_lp}")