"""
Created on Mon Dec  8 10:00:10 2025

@author: taha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_and_clean_data(filename, col_indices):
    """
    Charge et nettoie les données d'un fichier CSV.
    col_indices: liste des indices [Position Réelle, Vitesse, Ecart Type]
    """
    try:
        # Lecture brute
        df = pd.read_csv(filename, sep=',', dtype=str, encoding='latin1')
        
        # Sélection des colonnes
        data = df.iloc[:, col_indices].copy()
        data.columns = ['x_reel', 'vitesse', 'ecart_type']
        
        # Nettoyage formats numériques (virgule -> point)
        for col in data.columns:
            data[col] = pd.to_numeric(data[col].astype(str).str.replace(',', '.'), errors='coerce')
            
        # Suppression des NaNs et tri
        data = data.dropna()
        data = data.sort_values(by='x_reel')
        
        return data['x_reel'].values, data['vitesse'].values, data['ecart_type'].values
    except Exception as e:
        print(f"Erreur lors du chargement de {filename}: {e}")
        return None, None, None

def calculate_flow_rate(x_mm, u_ms, u_std, height_m=0.06):
    """Calcule le débit volumique et son incertitude (L/min)."""
    x_m = x_mm / 1000.0
    
    # Intégrales
    q_mean = np.trapz(u_ms, x_m) * height_m * 60000
    q_min = np.trapz(u_ms - u_std, x_m) * height_m * 60000
    q_max = np.trapz(u_ms + u_std, x_m) * height_m * 60000
    
    uncertainty = (q_max - q_min) / 2
    return q_mean, uncertainty

# --- CHARGEMENT DES DONNÉES ---

# Fichier 1 : Indices [5, 3, 4] (Position réelle, Vitesse, Ecart-type)
file1 = 'FLE 3 Banc hydraulique - LDA.csv'
x1, u1, std1 = load_and_clean_data(file1, [5, 3, 4])

# Fichier 2 : Indices [4, 2, 3] (Décalage d'un index dû au format du fichier)
file2 = 'FLE 3 Banc hydraulique - LDA 2.csv'
x2, u2, std2 = load_and_clean_data(file2, [4, 2, 3])

# --- CALCULS ---
q1, dq1 = calculate_flow_rate(x1, u1, std1)
q2, dq2 = calculate_flow_rate(x2, u2, std2)

# --- TRACÉ COMPARATIF ---
plt.figure(figsize=(12, 7))

# Jeu de données 1 (Bleu)
if x1 is not None:
    plt.fill_between(x1, u1 - std1, u1 + std1, color='grey', alpha=0.3)
    plt.errorbar(x1, u1, yerr=std1, fmt='-o', color='black', ecolor='black', 
                 capsize=4, label=f'LDA 1 (Q = {q1:.1f} $\pm$ {dq1:.1f} L/min)')

# Jeu de données 2 (Vert)
if x2 is not None:
    plt.fill_between(x2, u2 - std2, u2 + std2, color='lightgreen', alpha=0.3)
    plt.errorbar(x2, u2, yerr=std2, fmt='-s', color='green', ecolor='darkgreen', 
                 capsize=4, label=f'LDA 2 (Q = {q2:.1f} $\pm$ {dq2:.1f} L/min)')

plt.xlabel('$x$ (mm)', fontsize=12)
plt.ylabel('$U$ (m/s)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()

plt.show()