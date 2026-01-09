# -*- coding: utf-8 -*-
"""
Created on Fri Jan  9 12:55:18 2026

@author: taha
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
    Modèle de loi de puissance typique pour les déversoirs/hydraulique :
    Q = a * h^b
    """
    return a * (x**b)

# ---------------------------------------------------------
# 2. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ---------------------------------------------------------
filename = 'FLE 3 Banc hydraulique - Pelton.csv'

try:
    # Lecture du CSV (gestion de l'encodage et des séparateurs)
    df = pd.read_csv(filename, sep=None, engine='python', encoding='latin1')
    
    # Nettoyage des noms de colonnes (retirer espaces et accents bizarres)
    df.columns = [c.strip() for c in df.columns]
    
    # Extraction des colonnes utiles par position (plus robuste que le nom)
    # Col 0 : Hauteur d'eau (cm) -> h
    # Col 5 : Débit (L/min ou autre) -> Q
    raw_h = df.iloc[:, 0].astype(str).str.replace(',', '.')
    raw_Q = df.iloc[:, 5].astype(str).str.replace(',', '.')
    
    # Conversion en numérique
    x = pd.to_numeric(raw_h, errors='coerce').dropna().values
    y = pd.to_numeric(raw_Q, errors='coerce').dropna().values
    
    # On s'assure que x et y ont la même taille
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]

    print("Données chargées avec succès :")
    print(f"h (cm) : {x}")
    print(f"Q (L/min) : {y}")

except Exception as e:
    print(f"Erreur lors du chargement des données : {e}")
    # Données de secours (exemple) si le fichier n'est pas trouvé
    x = np.array([5.4, 6.2, 6.5, 7.3, 8.0, 8.5, 9.1, 9.5])
    y = np.array([14.7, 18.3, 21.1, 25.9, 29.9, 33.6, 37.1, 40.2])

# ---------------------------------------------------------
# 3. PARAMÈTRES D'INCERTITUDE (A ajuster selon votre TP)
# ---------------------------------------------------------
# Incertitude sur la hauteur h (ex: erreur de lecture règle)
# On suppose +/- 1 mm = 0.1 cm
deltax = np.array([0.1 for _ in x]) 

# Incertitude sur le débit Q
# On suppose une incertitude relative de 5% ou fixe (à affiner)
deltay = y * 0.05 

# ---------------------------------------------------------
# 4. ALGORITHME DE MONTE-CARLO
# ---------------------------------------------------------
plt.close('all')
fig, ax = plt.subplots(figsize=(10, 7))

N = int(1000) # Nombre de tirages
verbose = False

parama = [] # Stockage paramètre 'a'
paramb = [] # Stockage paramètre 'b' (l'exposant)

print(f"\nLancement de {N} tirages Monte-Carlo...")

for i in range(N):
    # Tirage aléatoire dans les barres d'erreur (Distribution Uniforme)
    xtemp = x + np.random.uniform(-deltax, deltax)
    ytemp = y + np.random.uniform(-deltay, deltay)
    
    try:
        # Ajustement du modèle de puissance
        # p0=[1, 1.5] sont des valeurs initiales pour aider l'algo (a=1, b=1.5)
        popt, pcov = curve_fit(modele_puissance, xtemp, ytemp, p0=[1, 1.5], maxfev=5000)
        
        parama.append(popt[0])
        paramb.append(popt[1])
        
        # Affichage des premières courbes (mode verbose)
        if verbose and i < 50:
            x_fit_plot = np.linspace(min(x), max(x), 100)
            y_fit_plot = modele_puissance(x_fit_plot, *popt)
            ax.plot(x_fit_plot, y_fit_plot, color="red", alpha=0.05)
            
    except RuntimeError:
        continue # Ignore les cas où le fit échoue

# ---------------------------------------------------------
# 5. ANALYSE STATISTIQUE DES RÉSULTATS
# ---------------------------------------------------------
# Moyennes et Écarts-types
a_mean = np.mean(parama)
b_mean = np.mean(paramb)
ua = np.std(parama)
ub = np.std(paramb)

print("\n" + "="*40)
print("RÉSULTATS DE L'ÉTALONNAGE : Q = a * h^b")
print("="*40)
print(f"Coefficient a = {a_mean:.4f} ± {2*ua:.4f} (95% conf)")
print(f"Exposant b    = {b_mean:.4f} ± {2*ub:.4f} (95% conf)")

# ---------------------------------------------------------
# 6. TRACÉ FINAL
# ---------------------------------------------------------

# Points expérimentaux
ax.errorbar(x, y, xerr=deltax, yerr=deltay, fmt='o', color='black', 
            ecolor='gray', capsize=3, label='Mesures expérimentales', zorder=5)

# Courbe moyenne
x_model = np.linspace(min(x)*0.9, max(x)*1.05, 200)
y_model = modele_puissance(x_model, a_mean, b_mean)
ax.plot(x_model, y_model, 'r-', linewidth=2, label=f'Modèle moyen : $Q = {a_mean:.2f} h^{{{b_mean:.2f}}}$')

# Intervalle de confiance (Enveloppe)
# On calcule l'enveloppe min et max basée sur les incertitudes de a et b
y_low = modele_puissance(x_model, a_mean - 2*ua, b_mean - 2*ub)
y_high = modele_puissance(x_model, a_mean + 2*ua, b_mean + 2*ub)
#ax.fill_between(x_model, y_low, y_high, color='red', alpha=0.2, label='Intervalle de confiance 95%')

# Mise en forme
ax.set_xlabel("Hauteur d'eau $h$ (cm)", fontsize=14)
ax.set_ylabel("Débit volumique $Q$ (L/min)", fontsize=14)
ax.set_title("Loi d'étalonnage Turbine Pelton (Monte Carlo)", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, which='both', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()