#!/usr/bin/env python3
import os
import time
import numpy as np
import scipy.sparse as spa
import csv
import burke_xu_lp as lp


""" Einstellungen """

loop = 1
verbose = False
acc = 1e-4
maxiter = 1000
crmaxiter = 100
sigma = 0.5
alpha_1 = 0.75
alpha_2 = 0.8
scaling = 0
presolve = (True, True, None)
data_path = "data/"


if loop == 1:

# CSV-Datei einlesen
    netlib_fun = {}
    with open(os.path.join(data_path, 'NetlibFun.csv'), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = row['npz_file']
            rows = int(row['Rows'])
            cols = int(row['Cols'])
            nonzeros = int(row['Nonzeros'])
            optimal_value = float(row['NetlibFun'])
            netlib_fun[filename] = {
                'Rows': rows,
                'Cols': cols,
                'Nonzeros': nonzeros,
                'Optimal Value': optimal_value
            }

    # Pfad zu den .npz-Dateien
    output_file = "testergebnisse.txt"
    latex_output = "latexergebnisse.txt"

    # Alle .npz-Dateien im Verzeichnis durchlaufen
    npz_files = sorted([f for f in os.listdir(data_path) if f.endswith('.npz')])

    # Datei einmalig öffnen und neu erstellen (alte wird gelöscht), externe Variablen und dann Kopfzeile schreiben
    with open(output_file, "w") as file:
        file.write(f"acc = {acc:.4E}\n")
        file.write(f"maxiter = {maxiter}\n")
        file.write(f"sigma = {sigma}\n")
        file.write(f"alpha1 = {alpha_1}\n")
        file.write(f"alpha2 = {alpha_2}\n")
        file.write(f"presolve = {presolve}\n")
        file.write(f"scaling = {scaling}\n")
        file.write("\n")
        file.write(f"{'Datei':<12}{'Rows':<10}{'Columns':<10}{'Nonzeros':<10}{'NetlibFun':<15}{'Fun':<15}{'mu':<10}{'phi':<10}{'Pred':<6}{'Iter':<6}{'Time(s)':<10}{'Status':<10}\n")
        file.write("="*120 + "\n")

    count = 0
    for npz_file in npz_files:
        presolve_m = presolve
        scaling_m = scaling
        npz_file_name = npz_file.rstrip('.npz')
        count += 1
        start_time = time.time()
        try:
            
            # Laden der Daten
            data = np.load(os.path.join(data_path, npz_file), allow_pickle=True)
            c = data["c"]
            A_eq = spa.csc_matrix(data["A_eq"])
            b_eq = data["b_eq"]
            A_ineq = spa.csc_matrix(data["A_ub"])
            b_ineq = data["b_ub"]
            bounds = np.squeeze(data["bounds"])
            print(f"Lade Datei {npz_file_name}. ({count}/{len(npz_files)})")

            # bei diesen LPs besondere Einstellung beim Preprocessing
            special_cases_red = ['BANDM', 'BORE3D', 'D2Q06C', 'GANGES', 'GREENBEA', 'GREENBEB', 'GROW15', 'GROW22', 'MAROS-R7', 'MAROS', 'PILOTNOV', 'QAP12', 'QAP15', 'SCSD8', 'WOOD1P', 'WOODW']
            if npz_file_name in special_cases_red:
                presolve_m = (True, False, None)
            special_cases_scaling2 = ['MODSZK1', 'PEROLD', 'PILOT', 'PILOT87', 'STOCFOR3']
            if npz_file_name in special_cases_scaling2:
                presolve_m = (True, False, None)
                scaling_m = 1
            no_presolve = ['PILOT-JA', 'BNL2']
            if npz_file_name in no_presolve:
                presolve_m = (False, False, None)

            # Standardausführung des Algorithmus
            start_time = time.time()
            try:
                result, slack, fun, nullstep, iter, exec_time = lp.burke_xu_lp(
                    c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, 
                    bounds=bounds, maxiter=maxiter, acc=acc, presolve=presolve_m, scaling=scaling_m, sigma=sigma, alpha_1=alpha_1, alpha_2=alpha_2, verbose=verbose
                )
                if np.isnan(fun[0]):
                    raise ValueError("Das Ergebnis 'Fun' ist NaN.")
                if iter == maxiter:
                    status = "Maxiter"
                else:
                    status = "Erfolg"
            except Exception as e:

                # Tritt ein Fehler auf: Ausführung des Algorithmus mit robusteren Einstellungen
                try:
                    presolve_m = (True, False, None)
                    result, slack, fun, nullstep, iter, exec_time = lp.burke_xu_lp(
                        c, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq, 
                        bounds=bounds, maxiter=crmaxiter, acc=1e-3, regularizer=1e-2, presolve=presolve_m, scaling=scaling, sigma=sigma, alpha_1=alpha_1, alpha_2=alpha_2, verbose=verbose
                    )
                    if iter == crmaxiter:
                        status = "CrMaxiter"
                    else:
                        status = "CrErfolg"
                except Exception as e2:
                    status = "Fehler"
                    result, slack, fun, nullstep, iter, exec_time = None, None, None, None, None, None
            netlib_values = netlib_fun[npz_file_name]
            rows = netlib_values['Rows']
            cols = netlib_values['Cols']
            netlibfun = netlib_values['Optimal Value']
            nullstep = iter - nullstep
            nonzeros = netlib_values['Nonzeros']

            # Testergebnisse öffnen und schreiben
            with open(output_file, "a") as file:
                if status == "Erfolg" or status == "Maxiter" or status == "CrErfolg" or status == "CrMaxiter":
                    file.write(f"{npz_file_name:<12}{rows:<10}{cols:<10}{nonzeros:<10}{netlibfun:<15.6E}{fun[0]:<15.6E}{fun[1]:<10.2E}{fun[2]:<10.2E}{nullstep:<6}{iter:<6}{exec_time:<10.2f}{status:<10}\n")
                else:
                    file.write(f"{npz_file_name:<12}{'-':<10}{'-':<10}{'-':<10}{'-':<15}{'-':<15}{'-':<10}{'-':<10}{'-':<6}{'-':<6}{'-':<10}{status:<10}\n")

            # Testergebnisse in Latex-Format für die .pdf schreiben
            with open(latex_output, "a") as file:
                if status in ["Erfolg", "Maxiter", "CrErfolg", "CrMaxiter"]:
                    file.write(f"{npz_file_name} &{nonzeros} &\\text{{{fun[0]:.4e}}} &\\text{{{fun[1]:.2e}}} &\\text{{{fun[2]:.2e}}} &{nullstep} &{iter} &{exec_time:.2f} &{status} \\\\ \n")
                else:
                    file.write(f"{npz_file_name} &{nonzeros} &- &- &- &- &- &- &Fehler \n")
        except Exception as e:

            # Fehler beim Laden der Datei oder beim Ausführen des Algorithmus
            with open(output_file, "a") as file:
                file.write(f"Fehler in {npz_file} \n")
            with open(latex_output, "a") as file:
                file.write(f"Fehler \n")
        end_time = time.time()
        print(f"{npz_file_name} führte nach {end_time - start_time:.2f} Sekunden zu Status '{status}'.")
        print(f"Nach Preprocessing und Überführung in Standardform benötigte der Algorithmus {exec_time:.2f} Sekunden für {iter} Iterationen.")

    print(f"Die Ergebnisse wurden in '{output_file}' gespeichert und in '{latex_output}' wurde passender Latex code geschrieben.")