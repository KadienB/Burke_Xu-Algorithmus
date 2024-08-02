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
    x = 0
    l = 0
    s = 0
    mu = 0
    if np.linalg.norm(mt.big_phi(x, s, 0)) < acc:
        maxiter = 0

    # Festsetzung der Stellschrauben
    beta = 0
    sigma = 0
    alpha1 = 0
    alpha2 = 0


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
        print(f"s^{k} = {s}")
        print(f"mu_{k} = {mu}")
        print(f"-----------------------------------------------------------")
        # Prädiktor-Schritt
        # Korrektor-Schritt


    """ Ausgabe des Ergebnisses """

    # Ausgabe des Ergebnisses


    return x, s