
"""
Created on Mon Dec  8 10:51:10 2025

@author: taha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. CHARGEMENT ET NETTOYAGE DES DONNÉES
# ---------------------------------------------------------
filename = 'FLE 3 Banc hydraulique - LDA 2.csv'

# Lecture du fichier (format CSV avec virgules comme séparateurs et décimales)
df = pd.read_csv(filename, sep=',', dtype=str, encoding='latin1')

# Sélection des colonnes par index (adapté à ce nouveau fichier)
# Index 4 : Position réelle
# Index 2 : Vitesse
# Index 3 : Ecart type
data = df.iloc[:, [4, 2, 3]].copy()
data.columns = ['x_reel', 'vitesse', 'ecart_type']

# --- Nettoyage des formats numériques ---
for col in data.columns:
    # Remplacement de la virgule décimale par un point et conversion
    data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '.'), errors='coerce')

# Suppression des lignes invalides et tri
data = data.dropna()
data = data.sort_values(by='x_reel')

# Variables pour le calcul
x_mm = data['x_reel'].values      # mm
u_ms = data['vitesse'].values     # m/s
u_std = data['ecart_type'].values # m/s

# ---------------------------------------------------------
# 2. CALCUL DU DÉBIT (Dimensions Veine : 60x60 mm)
# ---------------------------------------------------------
hauteur_veine_m = 60e-3
x_m = x_mm / 1000.0

# Intégration numérique (Trapèzes) pour le débit moyen
integrale_moyenne = np.trapz(u_ms, x_m)
Q_moyen_Lmin = (integrale_moyenne * hauteur_veine_m) * 60000

# Calcul de l'incertitude via les bornes min/max (U - std et U + std)
integrale_min = np.trapz(u_ms - u_std, x_m)
integrale_max = np.trapz(u_ms + u_std, x_m)

Q_min_Lmin = (integrale_min * hauteur_veine_m) * 60000
Q_max_Lmin = (integrale_max * hauteur_veine_m) * 60000

incertitude = (Q_max_Lmin - Q_min_Lmin) / 2

print("="*40)
print(f"FICHIER : {filename}")
print(f"Débit Moyen       : {Q_moyen_Lmin:.2f} L/min")
print(f"Incertitude (+/-) : {incertitude:.2f} L/min")
print("="*40)

# ---------------------------------------------------------
# 3. TRACÉ
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

# Zone d'incertitude (bande grise)
plt.fill_between(x_mm, u_ms - u_std, u_ms + u_std, color='gray', alpha=0.2, label='Incertitude ($\pm \sigma$)')

# Courbe et points de mesure
plt.errorbar(x_mm, u_ms, yerr=u_std, fmt='-o', color='darkgreen', ecolor='orange', capsize=4, label='Mesures LDA 2')

plt.xlabel('Position horizontale $x$ (mm)')
plt.ylabel('Vitesse $U$ (m/s)')
plt.title(f'Profil de vitesse (Fichier 2) et débit estimé\nQ = {Q_moyen_Lmin:.1f} $\pm$ {incertitude:.1f} L/min')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()