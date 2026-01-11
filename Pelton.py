# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 12:55:18 2026

@author: taha
Script corrigé pour unités SI : h en [m] et Q en [m^3/s]
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------
# 1. DÉFINITION DU MODÈLE
# ---------------------------------------------------------
def modele_puissance(x, a, b):
    """
    Modèle : Q = a * h^b
    """
    return a * (x**b)

# ---------------------------------------------------------
# 2. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ---------------------------------------------------------
filename = 'FLE 3 Banc hydraulique - Pelton.csv'

try:
    # Lecture du CSV
    df = pd.read_csv(filename, sep=None, engine='python', encoding='latin1')
    
    # Nettoyage colonnes
    df.columns = [c.strip() for c in df.columns]
    
    # Extraction des colonnes (On suppose que le fichier contient bien les valeurs en m et m^3/s)
    # Col 0 : Hauteur d'eau h (m)
    # Col 5 : Débit Q (m^3/s)
    raw_h = df.iloc[:, 0].astype(str).str.replace(',', '.')
    raw_Q = df.iloc[:, 5].astype(str).str.replace(',', '.')
    
    # Conversion
    x = pd.to_numeric(raw_h, errors='coerce').dropna().values
    y = pd.to_numeric(raw_Q, errors='coerce').dropna().values
    
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    print("Données chargées (unités SI supposées) :")
    print(f"h [m]      : {x[:5]} ...")
    print(f"Q [m^3/s]  : {y[:5]} ...")

except Exception as e:
    print(f"Erreur lors du chargement : {e}")
    # Valeurs de secours (converties en SI pour l'exemple)
    x = np.array([0.054, 0.062, 0.065, 0.073, 0.080, 0.085, 0.091, 0.095])
    y = np.array([14.7, 18.3, 21.1, 25.9, 29.9, 33.6, 37.1, 40.2]) / 60000.0 # L/min -> m3/s

# ---------------------------------------------------------
# 3. PARAMÈTRES D'INCERTITUDE
# ---------------------------------------------------------
# Incertitude sur h : +/- 1 mm = 0.001 m
deltax = np.array([0.001 for _ in x]) 

# Incertitude sur Q : 5% relatif (classique)
deltay = y * 0.05 

# ---------------------------------------------------------
# 4. ALGORITHME DE MONTE-CARLO
# ---------------------------------------------------------
plt.close('all')
fig, ax = plt.subplots(figsize=(10, 7))

N = int(1000) 
parama = [] 
paramb = [] 

print(f"\nLancement de {N} tirages Monte-Carlo...")

for i in range(N):
    # Tirage aléatoire
    xtemp = x + np.random.uniform(-deltax, deltax)
    ytemp = y + np.random.uniform(-deltay, deltay)
    
    try:
        # Ajustement
        # p0=[0.5, 2.0] : On aide l'algo. 'a' sera petit (~0.5 ou moins) et 'b' autour de 1.5-2.5
        popt, pcov = curve_fit(modele_puissance, xtemp, ytemp, p0=[0.5, 2.0], maxfev=5000)
        
        parama.append(popt[0])
        paramb.append(popt[1])
        
    except RuntimeError:
        continue 

# ---------------------------------------------------------
# 5. ANALYSE STATISTIQUE
# ---------------------------------------------------------
a_mean = np.mean(parama)
b_mean = np.mean(paramb)
ua = np.std(parama)
ub = np.std(paramb)

print("\n" + "="*40)
print("RÉSULTATS ÉTALONNAGE (Unités SI)")
print("="*40)
print(f"Coefficient a = {a_mean:.6f} ± {2*ua:.6f} (95%)")
print(f"Exposant b    = {b_mean:.4f} ± {2*ub:.4f} (95%)")

# ---------------------------------------------------------
# 6. TRACÉ
# ---------------------------------------------------------

# Points expérimentaux
ax.errorbar(x, y, xerr=deltax, yerr=deltay, fmt='o', color='black', 
            ecolor='gray', capsize=3, label='Mesures expérimentales', zorder=5)

# Courbe moyenne
# On étend un peu la plage pour le tracé
x_model = np.linspace(min(x)*0.9, max(x)*1.05, 200)
y_model = modele_puissance(x_model, a_mean, b_mean)

ax.plot(x_model, y_model, 'r-', linewidth=2, label=f'Modèle : $Q = {a_mean:.3f} h^{{{b_mean:.2f}}}$')

# Intervalle de confiance
y_low = modele_puissance(x_model, a_mean - 2*ua, b_mean - 2*ub)
y_high = modele_puissance(x_model, a_mean + 2*ua, b_mean + 2*ub)
ax.fill_between(x_model, y_low, y_high, color='red', alpha=0.15, label='Confiance 95%')

# Mise en forme (Labels corrigés)
ax.set_xlabel("Hauteur d'eau $h$ (m)", fontsize=14)
ax.set_ylabel("Débit volumique $Q$ ($m^3/s$)", fontsize=14)
ax.set_title("Loi d'étalonnage Turbine Pelton ($Q=ah^b$)", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--', alpha=0.6)

# Utilisation de la notation scientifique pour l'axe Y (car débits petits)
ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))

plt.tight_layout()
plt.show()