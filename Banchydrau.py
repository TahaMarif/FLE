#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# =============================================================================
# 1. CONFIGURATION ET PARAMÈTRES D'Incertitude
# =============================================================================

# Choix du capteur à analyser : 'ROTAMETRE', 'VENTURI' ou 'DIAPHRAGME'
CAPTEUR = 'DIAPHRAGME'  # Modifier selon le capteur utilisé

# Estimation des incertitudes de lecture (à modifier selon votre matériel)
INCERTITUDE_H_MM = 2.0      # Incertitude de lecture sur la règle (en mm), ex: +/- 1mm donc intervalle de 2
INCERTITUDE_DEBIT_REL = 0.05 # Incertitude relative sur le débit (5% pour chronomètre + balance)

# Paramètres physiques
RHO = 1000  # kg/m3
G = 9.81    # m/s2

# =============================================================================
# 2. CHARGEMENT ET PRÉPARATION DES DONNÉES
# =============================================================================

fichier_csv = 'FLE 3 Banc hydraulique - Feuille 1.csv'

# Lecture robuste du CSV
try:
    df = pd.read_csv(fichier_csv, decimal=',')
except:
    df = pd.read_csv(fichier_csv)
    # Nettoyage si nécessaire
    cols = ['Débit en m3/s', 'Hauteur Rotamètre en mm', 'DeltaHAB', 'DeltaH_EF']
    for col in cols:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

# Récupération du débit (Y) et calcul de son incertitude (Delta Y)
y_raw = df['Débit en m3/s'].values
# On supprime les NaN éventuels
mask = ~np.isnan(y_raw)
y = y_raw[mask]

# Calcul de l'incertitude absolue sur le débit (ici 5% de la valeur mesurée)
deltay = y * INCERTITUDE_DEBIT_REL

# Préparation des X selon le capteur choisi
if CAPTEUR == 'ROTAMETRE':
    # X = Hauteur en mm
    x_raw = df['Hauteur Rotamètre en mm'].values[mask]
    deltax = np.full(len(x_raw), INCERTITUDE_H_MM) # Erreur constante de lecture
    
    xlabel = "Hauteur (mm)"
    titre_graphe = "Étalonnage Rotamètre (Monte-Carlo)"
    
    # Modèle affine (Q = aH + b)
    def modele(x, a, b):
        return a * x + b
    p0 = [1e-5, 0] # estimation initiale pour aider l'algo

elif CAPTEUR in ['VENTURI', 'DIAPHRAGME']:
    # Sélection de la colonne
    col_name = 'DeltaHAB' if CAPTEUR == 'VENTURI' else 'DeltaH_EF'
    h_mm = df[col_name].values[mask]
    
    # Conversion en Pression (Pa) : P = rho * g * h(m)
    delta_P = RHO * G * (h_mm * 1e-3)
    
    # X = Racine carrée de la pression (pour avoir une droite)
    x = np.sqrt(np.abs(delta_P))
    
    # PROPAGATION D'INCERTITUDE complexe :
    # Si H a une incertitude delta_H, quelle est l'incertitude sur sqrt(rho*g*H) ?
    # Calcul par dérivée : d(sqrt(C*H)) = 0.5 * sqrt(C/H) * dH
    # C = rho * g * 1e-3
    C = RHO * G * 1e-3
    deltax = 0.5 * np.sqrt(C / (h_mm + 1e-9)) * INCERTITUDE_H_MM
    
    xlabel = r"$\sqrt{\Delta P}$ ($Pa^{1/2}$)"
    titre_graphe = f"Étalonnage {CAPTEUR} (Loi en racine)"
    
    # Modèle Linéaire passant par zéro (Q = K * X) car si DeltaP=0, Q=0
    def modele(x, k):
        return k * x
    p0 = [1e-4] # estimation initiale

else:
    raise ValueError("Capteur non reconnu. Choisissez ROTAMETRE, VENTURI ou DIAPHRAGME")

# Si on est en mode Rotamètre, x est assigné ici (pour Venturi c'est déjà fait plus haut)
if CAPTEUR == 'ROTAMETRE':
    x = x_raw

# =============================================================================
# 3. ALGORITHME DE MONTE-CARLO (Cœur du script)
# =============================================================================

plt.close('all')
fig, ax = plt.subplots(figsize=(12, 8))

N = 1000 # Nombre de tirages
params_stock = [] # Stockage des résultats

# Génération des axes pour le tracé lisse
x_fit_plot = np.linspace(0, np.max(x)*1.1, 1000)

for i in range(N):
    # 1. Tirage aléatoire des données expérimentales dans leur intervalle d'incertitude
    # Distribution uniforme (rectangulaire)
    xtemp = [val + np.random.uniform(-err, err) for val, err in zip(x, deltax)]
    ytemp = [val + np.random.uniform(-err, err) for val, err in zip(y, deltay)]
    
    # 2. Ajustement du modèle sur ces données bruitées
    try:
        if CAPTEUR == 'ROTAMETRE':
            popt, _ = curve_fit(modele, xtemp, ytemp, p0=p0)
            params_stock.append(popt) # stocke [a, b]
            
            # Affichage visuel (optionnel, ralentit si N grand)
            if i < 50: # On n'affiche que les 50 premiers pour ne pas surcharger
                y_fit_temp = modele(x_fit_plot, *popt)
                ax.plot(x_fit_plot, y_fit_temp, color="red", alpha=0.05)
                
        else: # Venturi / Diaph
            popt, _ = curve_fit(modele, xtemp, ytemp) # pas besoin de p0 souvent
            params_stock.append(popt[0]) # stocke k
            
            if i < 50:
                y_fit_temp = modele(x_fit_plot, popt[0])
                ax.plot(x_fit_plot, y_fit_temp, color="red", alpha=0.05)

    except RuntimeError:
        pass # Ignore les fits qui échouent

# =============================================================================
# 4. ANALYSE STATISTIQUE ET AFFICHAGE
# =============================================================================

params_stock = np.array(params_stock)

print("-" * 50)
print(f"RÉSULTATS POUR : {CAPTEUR}")
print("-" * 50)

if CAPTEUR == 'ROTAMETRE':
    # Récupération des distributions de a et b
    a_vals = params_stock[:, 0]
    b_vals = params_stock[:, 1]
    
    a_mean, a_std = np.mean(a_vals), np.std(a_vals)
    b_mean, b_std = np.mean(b_vals), np.std(b_vals)
    
    # Tracé de la courbe moyenne
    y_fit_mean = modele(x_fit_plot, a_mean, b_mean)
    ax.plot(x_fit_plot, y_fit_mean, color="red", linewidth=2, label="Ajustement moyen")
    
    # Zone de confiance (approximative)
    y_upper = modele(x_fit_plot, a_mean + 2*a_std, b_mean + 2*b_std)
    y_lower = modele(x_fit_plot, a_mean - 2*a_std, b_mean - 2*b_std)
    ax.fill_between(x_fit_plot, y_lower, y_upper, color="red", alpha=0.2, label="Intervalle confiance 95%")
    
    print(f"Modèle : Q = a * H + b")
    print(f"a = {a_mean:.3e} ± {2*a_std:.3e} (m^2/s)")
    print(f"b = {b_mean:.3e} ± {2*b_std:.3e} (m^3/s)")

else: # Venturi / Diaph
    k_vals = params_stock # tableau 1D
    
    k_mean, k_std = np.mean(k_vals), np.std(k_vals)
    
    # Tracé moyen
    y_fit_mean = modele(x_fit_plot, k_mean)
    ax.plot(x_fit_plot, y_fit_mean, color="red", linewidth=2, label="Ajustement moyen")
    
    # Zone de confiance
    y_upper = modele(x_fit_plot, k_mean + 2*k_std)
    y_lower = modele(x_fit_plot, k_mean - 2*k_std)
    ax.fill_between(x_fit_plot, y_lower, y_upper, color="red", alpha=0.2, label="Intervalle confiance 95%")
    
    print(f"Modèle : Q = K * sqrt(DeltaP)")
    print(f"K = {k_mean:.3e} ± {2*k_std:.3e}")


# Affichage des points expérimentaux avec barres d'erreur
ax.errorbar(x, y, xerr=deltax, yerr=deltay, fmt='o', color='black', 
            ecolor='black', capsize=3, label='Mesures Expérimentales')

# Esthétique
ax.legend(fontsize=12)
ax.set_xlabel(xlabel, fontsize=14)
ax.set_ylabel("Débit volumique ($m^3/s$)", fontsize=14)
ax.set_title(titre_graphe, fontsize=16)
ax.grid(True, which='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
# plt.savefig(f"Resultat_{CAPTEUR}.png")