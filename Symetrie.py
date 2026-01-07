# -*- coding: utf-8 -*-
"""
Created on Mon Dec 8 11:00:56 2025

@author: taha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_data(filename, col_indices):
    """Charge et nettoie les données CSV."""
    try:
        df = pd.read_csv(filename, sep=',', dtype=str, encoding='latin1')
        data = df.iloc[:, col_indices].copy()
        data.columns = ['x_reel', 'vitesse', 'ecart_type']
        for col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '.'), errors='coerce')
        return data.dropna().sort_values(by='x_reel')
    except Exception as e:
        print(f"Erreur {filename}: {e}")
        return None

def calculate_flow_symmetry(df, label, ax=None, L=60):
    """
    Calcule le débit par symétrie radiale et affiche le résultat sur l'axe donné.
    ax: objet matplotlib axes (optionnel)
    """
    if df is None: return 0, 0
    
    center = L / 2  # Centre à 30 mm
    
    # 1. Conversion en coordonnées radiales
    df['r'] = np.abs(df['x_reel'] - center)
    df_r = df.sort_values(by='r')
    
    r_vals = df_r['r'].values
    u_vals = df_r['vitesse'].values
    std_vals = df_r['ecart_type'].values
    
    # Ajout point zéro paroi
    if r_vals.max() < (center - 0.5):
         r_vals = np.append(r_vals, center)
         u_vals = np.append(u_vals, 0.0)
         std_vals = np.append(std_vals, 0.0)
    
    # Interpolation
    f_u = interp1d(r_vals, u_vals, kind='linear', bounds_error=False, fill_value=(u_vals[0], 0))
    f_std = interp1d(r_vals, std_vals, kind='linear', bounds_error=False, fill_value=(std_vals[0], 0))
    
    # 2. Grille 2D
    grid_step = 0.5 
    x_grid = np.arange(-center, center + grid_step, grid_step)
    y_grid = np.arange(-center, center + grid_step, grid_step)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    rr = np.sqrt(xx**2 + yy**2)
    
    # 3. Application symétrie + Coins morts
    U_map = f_u(rr)
    U_std_map = f_std(rr)
    
    U_map[rr > center] = 0 
    U_std_map[rr > center] = 0
    
    # 4. Calcul Débit
    dA = (grid_step * 1e-3)**2 
    Q_m3s = np.sum(U_map) * dA
    Q_Lmin = Q_m3s * 60000 
    
    Q_max = np.sum(U_map + U_std_map) * dA * 60000
    Q_min = np.sum(U_map - U_std_map) * dA * 60000
    uncert = (Q_max - Q_min) / 2
    
    # --- AFFICHAGE SUR L'AXE ---
    if ax is None:
        fig, ax = plt.subplots()
    
    # Tracé de la heatmap
    im = ax.pcolormesh(xx, yy, U_map, shading='auto', cmap='jet', vmin=0)
    
    # Cercle inscrit pour visualisation
    circle = plt.Circle((0, 0), center, color='white', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    ax.set_title(f"{label}\nQ = {Q_Lmin:.2f} $\pm$ {uncert:.2f} L/min")
    ax.set_aspect('equal')
    ax.set_xlabel('x (mm)')
    if label == "LDA 1": # On met le label Y seulement sur le premier
        ax.set_ylabel('y (mm)')
    
    # On retourne l'objet image 'im' pour pouvoir faire une barre de couleur commune plus tard si besoin
    # Mais ici on va ajouter une colorbar individuelle pour chaque plot pour la lisibilité
    plt.colorbar(im, ax=ax, label='Vitesse (m/s)', fraction=0.046, pad=0.04)
    
    return Q_Lmin, uncert

# --- EXÉCUTION ---
# Chargement
df1 = load_data('FLE 3 Banc hydraulique - LDA.csv', [5, 3, 4])
df2 = load_data('FLE 3 Banc hydraulique - LDA 2.csv', [4, 2, 3])

# Création de la figure globale (1 ligne, 2 colonnes)
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

print("--- RÉSULTATS COMPARATIFS ---")

# Calcul et tracé sur la gauche (axs[0])
Q1, dQ1 = calculate_flow_symmetry(df1, "LDA 1", ax=axs[0])
print(f"LDA 1 (Symétrie) : {Q1:.2f} ± {dQ1:.2f} L/min")

# Calcul et tracé sur la droite (axs[1])
Q2, dQ2 = calculate_flow_symmetry(df2, "LDA 2", ax=axs[1])
print(f"LDA 2 (Symétrie) : {Q2:.2f} ± {dQ2:.2f} L/min")

# Titre global et ajustement
plt.suptitle("Reconstruction de la section par Symétrie Centrale", fontsize=14)
plt.tight_layout()
plt.show()