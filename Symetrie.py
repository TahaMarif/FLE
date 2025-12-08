# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 11:00:56 2025

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

def calculate_flow_symmetry(df, label, L=60):
    """Calcule le débit en reconstruisant la section 2D par symétrie radiale."""
    if df is None: return
    
    center = L / 2  # Centre à 30 mm
    
    # 1. Conversion en coordonnées radiales (distance au centre)
    df['r'] = np.abs(df['x_reel'] - center)
    df_r = df.sort_values(by='r')
    
    r_vals = df_r['r'].values
    u_vals = df_r['vitesse'].values
    std_vals = df_r['ecart_type'].values
    
    # Ajout d'un point zéro à la paroi (r=30) si les données s'arrêtent avant
    if r_vals.max() < (center - 0.5):
         r_vals = np.append(r_vals, center)
         u_vals = np.append(u_vals, 0.0)
         std_vals = np.append(std_vals, 0.0)
    
    # Création des fonctions d'interpolation U(r) et sigma(r)
    # fill_value=(u_vals[0], 0) : On prolonge la valeur max au centre (r=0), et 0 à l'extérieur
    f_u = interp1d(r_vals, u_vals, kind='linear', bounds_error=False, fill_value=(u_vals[0], 0))
    f_std = interp1d(r_vals, std_vals, kind='linear', bounds_error=False, fill_value=(std_vals[0], 0))
    
    # 2. Création d'une grille 2D représentant la section carrée
    grid_step = 0.5 # Résolution de 0.5 mm
    x_grid = np.arange(-center, center + grid_step, grid_step) # de -30 à +30
    y_grid = np.arange(-center, center + grid_step, grid_step)
    xx, yy = np.meshgrid(x_grid, y_grid)
    
    # Calcul du rayon pour chaque point de la grille
    rr = np.sqrt(xx**2 + yy**2)
    
    # 3. Application de la symétrie centrale
    U_map = f_u(rr)
    U_std_map = f_std(rr)
    
    # HYPOTHÈSE FORTE : Recirculation / Zone morte dans les coins
    # Si r > 30 mm (cercle inscrit), on force la vitesse à 0
    U_map[rr > center] = 0 
    U_std_map[rr > center] = 0
    
    # 4. Intégration (Somme des pixels * Surface pixel)
    dA = (grid_step * 1e-3)**2 # Surface d'un point en m²
    
    Q_m3s = np.sum(U_map) * dA
    Q_Lmin = Q_m3s * 60000 # Conversion en L/min
    
    # Calcul incertitude (somme des variances ou bornes min/max)
    # Ici bornes min/max pour rester cohérent avec l'approche précédente
    Q_max = np.sum(U_map + U_std_map) * dA * 60000
    Q_min = np.sum(U_map - U_std_map) * dA * 60000
    uncert = (Q_max - Q_min) / 2
    
    # Affichage Carte
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(xx, yy, U_map, shading='auto', cmap='jet')
    plt.colorbar(label='Vitesse (m/s)')
    plt.title(f"Reconstruction {label}\n(Symétrie Centrale, Coins=0)\nQ = {Q_Lmin:.2f} L/min")
    plt.axis('equal')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.tight_layout()
    plt.show()
    
    return Q_Lmin, uncert

# --- EXÉCUTION ---
# Chargement des fichiers (avec les indices de colonnes corrects)
df1 = load_data('FLE 3 Banc hydraulique - LDA.csv', [5, 3, 4])
df2 = load_data('FLE 3 Banc hydraulique - LDA 2.csv', [4, 2, 3])

print("--- RÉSULTATS COMPARATIFS ---")

# Calcul LDA 1
Q1, dQ1 = calculate_flow_symmetry(df1, "LDA 1")
print(f"LDA 1 (Symétrie) : {Q1:.2f} ± {dQ1:.2f} L/min")

# Calcul LDA 2
Q2, dQ2 = calculate_flow_symmetry(df2, "LDA 2")
print(f"LDA 2 (Symétrie) : {Q2:.2f} ± {dQ2:.2f} L/min")