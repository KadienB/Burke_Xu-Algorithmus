from typing import Optional, Union, Tuple
import time
import numpy as np
import scipy.sparse as spa
import sksparse.cholmod as cholmod


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
        in der Regel aktueller s-Vektor in |R^n.
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

    # Brechnung von phi(a,b,mu)
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
    r"""Berechnung der Teilblockmatrizen von nabla(x,y)F wie in [Anmerkung (Effizientes Lösen der linearen Gleichungssysteme)] beschrieben unter der Verwendung von

    partial phi / partial a = 1 - (a - b) / sqrt((a - b)^2 + 4mu^2).

    partial phi / partial b = 1 + (a - b) / sqrt((a - b)^2 + 4mu^2).

    partial phi / partial mu = (-4mu) / sqrt((a - b)^2 + 4mu^2).

    Gespeichert wird jeweils in Vektorform, da es sich um vollbeschriebene Diagonalmatrizen handelt.

    Robustes Berechnen der Matrix (D_x_inv^(-1)D_s)^(1/2) als sparse Diagonalmatrix zur effizienten Verwendung mit scikit-sparse_AAt.

    Parameter
    ----------
    a: 
        in der Regel aktueller x-Vektor in |R^n.
    b:
        in der Regel aktueller s-Vektor in |R^n.
    mu:
        Glättungsparameter mu > 0.
    arg:
        Integer, der aussagt nach welchem Argument abgeleitet wird.
    inv:
        Boolean Variable um eine effizient und stabil berechnete Inverse zu erzeugen.
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

    # Berechnung von (D_x_inv^(-1)D_s)^(1/2)
    if arg == 0:
        sqrt_term = np.sqrt((a - b) ** 2 + (4 * mu**2))
        denominator_x_inv = np.maximum(- a + b + sqrt_term, bound)
        nabla_x_inv = np.sqrt(sqrt_term / denominator_x_inv)
        nabla_s = np.maximum(np.sqrt(1 + ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))), bound)
        diag = nabla_x_inv * nabla_s
        solution = spa.diags(diag, format='csc')

    elif arg == 1:
        # Berechnung von D_x
        if inv == False:
            solution = 1 - ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))
        # Berechnung von D_x_inv
        else:
            sqrt_term = np.sqrt((a - b) ** 2 + (4 * mu**2))
            denominator_x_inv = np.maximum(- a + b + sqrt_term, bound)
            solution = sqrt_term / denominator_x_inv
    elif arg == 2:
        # Berechnung von D_s
        if inv == False:
            solution = 1 + ((a - b) / np.sqrt((a - b) ** 2 + 4 * mu ** 2))
        # Berechnung von D_s_inv (wird quasi nicht verwendet, nur der Vollständigkeit wegen)
        else:
            sqrt_term = np.sqrt((a - b) ** 2 + (4 * mu**2))
            solution = sqrt_term / ((a - b) + sqrt_term)
        # Berechnung von D_mu
    elif arg == 3: 
        solution = (-4 * mu) / np.sqrt((a - b) ** 2 + 4 * mu ** 2)
    else: 
        raise ValueError("Argument must be 0, 1, 2 or 3.")

    if verbose:
        print(f"nabla_big_phi result = {solution}")

    return solution

def cholesky_decomposition_lhs(
    x: Optional[np.ndarray] = None,
    a: Optional[np.ndarray] = None,
    mu: Optional[float] = None,
    A: Union[np.ndarray, spa.csc_matrix] = None,
    use_sparse: bool = False,
    factor: Optional[cholmod.Factor] = None,
    regularizer: Optional[float] = 0,
    verbose: bool = False,    
)   -> Optional[Tuple[np.ndarray, np.ndarray]]:
    r"""Beschreibung

    Effiziente und robuste Implementation von sci-kit sparse für unseren Algorithmus.

    Parameter
    ----------
    x: 
        in der Regel aktueller x-Vektor in |R^n.
    a:
        in der Regel aktueller s-Vektor in |R^n.
    mu:
        Glättungsparameter.
    A:
        Matrix A für das Gleichungssystem A(D_x_inv D_s)A^T.
    use_sparse: 
        Boolean Variable, die aussagt, ob es sich um csc_matrix handelt. (Da dense Matrizen nie Effizient implementiert wurden mittlerweile obsolet.)
    factor:
        Vorherige Cholesky-Zerlegung, die über Informationen zur Struktur von A und damit von A((D_x_inv D_s)^(1/2)) verfügt.
    regularizer :
        Vorgegebener Vorfaktor der auf das Gleichungssystem addierten Einheitsmatrix. Standardmäßig wird reguralizer = acc^2 verwendet.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """

    max_regularizer = 1e-2

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
                while regularizer <= max_regularizer:
                    try:
                        # Versuche die Berechnung mit dem aktuellen Regularizer
                        factor = cholmod.cholesky_AAt(A, beta=regularizer)
                        if verbose:
                            print(f"Erfolgreich mit Regularizer = {regularizer}")
                        break  # Verlasse die Schleife, wenn erfolgreich
                    except Exception as e:
                        if verbose:
                            print(f"Fehler bei der Berechnung mit beta={regularizer}: {e}")
                        # Erhöhe den Regularizer um den Faktor 10
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
                while regularizer <= max_regularizer:
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

    Backtracking-Routine des Prädiktor-Schritts inklusive Abbruchkriterium. Die resultierenden Iterationsvektoren werden zurückgegeben.

    Parameter
    ----------
    x: 
        in der Regel aktueller x-Vektor in |R^n.
    a:
        in der Regel aktueller s-Vektor in |R^n.
    b:
        in der Regel aktueller lambda-Vektor in |R^m.
    delta_x
        in der Regel aktuelle Schritteweite Delta_x in |R^n.
    delta_a
        in der Regel aktuelle Schritteweite Delta_s in |R^n.
    delta_b
        in der Regel aktuelle Schritteweite Delta_lambda in |R^m.
    mu:
        Glättungsparameter.
    alpha_1:
        Externer Parameter für die Backtracking Routine.
    beta:
        Externer Parameter für die Beschränkung der Umgebung.
    acc:
        Gewünschte Genauigkeit für das Abbruch-Kriterium.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    
    if verbose:
        print(f"Starting predictor_step calculation...")

    # Überprüfung, ob der Prädiktor-Schritt das LP schon löst
    if np.linalg.norm(big_phi(x + delta_x, a + delta_a, 0)) < acc:
        step = 1
        x = x + delta_x
        a = a + delta_a
        if b is not None:
            b = b + delta_b

    # Überprüfung, ob der Prädiktor-Schritt in der gewünschten Umgebung endet
    elif np.linalg.norm(big_phi(x + delta_x, a + delta_a, mu, verbose=verbose)) > beta * mu:
        step = 0
        if verbose:
            print(f"Prädiktor-Schritt abgelehnt, da {np.linalg.norm(big_phi(x + delta_x, a + delta_a, mu, verbose=verbose))} > {beta * mu}")
            print(f"War der vorherige Wert in der Umgebung? {np.linalg.norm(big_phi(x , a, mu, verbose=verbose))} <= {beta * mu}")

    # Backtracking Routine, um zu schauen wie weit mu reduziert werden kann
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

    Backtracking-Routine des Korrektor-Schritts. Die resultierenden Iterationsvektoren werden zurückgegeben.

    Parameter
    ----------
    x: 
        in der Regel aktueller x-Vektor in |R^n.
    a:
        in der Regel aktueller s-Vektor in |R^n.
    b:
        in der Regel aktueller lambda-Vektor in |R^m.
    delta_x
        in der Regel aktuelle Schritteweite Delta_x in |R^n.
    delta_a
        in der Regel aktuelle Schritteweite Delta_s in |R^n.
    delta_b
        in der Regel aktuelle Schritteweite Delta_lambda in |R^m.
    mu:
        Glättungsparameter.
    alpha_2:
        Externer Parameter für die Backtracking Routine.
    beta:
        Externer Parameter für die Beschränkung der Umgebung.
    sigma:
        Externer Zentrierungsparameter.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    
    if verbose:
        print(f"Starting corrector_step calculation...")
    # Backtracking Routine des Korrektor-Schritts zur Bestimmung der Schrittweite
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
        if A_eq is not None:
            A_eq = A_eq.astype(np.float64)
        if A_ineq is not None:
            A_ineq = A_ineq.astype(np.float64)
    if b_eq is not None:
        b_eq = b_eq.astype(np.float64)
    if b_ineq is not None:
        b_ineq = b_ineq.astype(np.float64)
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

    Parameter
    ----------
    x_std: 
        Lösungsvektor des Linearen Programms nach Preprocessing.
    transformations:
        Dictionary mit den, beim auf Standardform bringen, gemachten Transformationen von A_std.
    initial_length:
        Länge des Lösungsvektors des Ausgangsform.
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    
    if verbose:
        print(f"Starting standardform_to_lp...")

    # transformations dictionary durchgehen und nach Fallunterscheidung Umwandlungen rückgängig machen
    for key in transformations.keys():

        # Fall 2:
        if transformations[key][0] == 2:
            x_std[key] -= x_std[transformations[key][1]]

        # Fall 3:
        elif transformations[key][0] == 3:
            x_std[key] += transformations[key][1]

        # Fall 5:
        elif transformations[key][0] == 5:
            x_std[key] += transformations[key][1]
            x_std[key] = -x_std[key]

        # Fall 6:
        elif transformations[key][0] == 6:
            x_std[key] += transformations[key][1]

    slack = x_std[initial_length:]
    x_std = x_std[:initial_length]

    return x_std, slack