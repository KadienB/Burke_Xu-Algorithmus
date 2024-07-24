import os
from typing import Optional, Iterator, Union
import numpy as np
import qpbenchmark
import scipy.io as spio
import scipy.sparse as spa
from qpbenchmark.benchmark import main
from qpbenchmark.problem import Problem
from qpsolvers import solve_qp

def small_phi(
    a: float,
    b: float,
    mu: float,
    verbose: bool = False,
)   -> Optional[float]:
    r"""Chen-Harker-Kanzow Glättungsmethode aus Kapitel 1.2 in der Thesis.

    Die Berechnung hat dabei die Form

    a + b - sqrt((a-b)^2 + 4mu^2)

    Parameters
    ----------
    a: 
        in der Regel Komponente des aktuellen x-Vektors.
    b:
        in der Regel Komponente des aktuellen y-Vektors.
    mu:
        Glättungsparameter
    verbose: bool
        Boolean Variable um eine Ausgabe sämtlicher Zwischenergebnisse zu erzeugen.
    """
    return

def big_phi(
    a: float,
    b: float,
    mu: float,
    verbose: bool = False,
):
    print("Diese Methode kommt aus der anderen Datei!")

def delta_x_big_phi(
    a: float,
    b: float,
    mu: float,
    verbose: bool = False,
):
    print("Diese Methode kommt aus der anderen Datei!")

def delta_y_big_phi(
    a: float,
    b: float,
    mu: float,
    verbose: bool = False,
):
    print("Diese Methode kommt aus der anderen Datei!")

def delta_mu_big_phi(
    a: float,
    b: float,
    mu: float,
    verbose: bool = False,
):
    print("Diese Methode kommt aus der anderen Datei!")

def predictor_step():
    print("Diese Methode kommt aus der anderen Datei!")

def corrector_step():
    print("Diese Methode kommt aus der anderen Datei!")