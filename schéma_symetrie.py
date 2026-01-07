# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 11:44:18 2026

@author: taha
"""
import numpy as np
import matplotlib.pyplot as plt

# Création de la grille (Veine carrée 60x60mm)
L = 60
x = np.linspace(-L/2, L/2, 200)
y = np.linspace(-L/2, L/2, 200)
X, Y = np.meshgrid(x, y)

# Profil de vitesse théorique (parabole simple pour l'exemple)
# U_max * (1 - (r/R)^2)
U_max = 1.0
R = L/2

# --- MODELE 1 : EXTRUSION (Uniforme verticalement) ---
# La vitesse ne dépend que de la position horizontale X (valeur absolue pour symétrie gauche/droite)
U_extrusion = U_max * (1 - (np.abs(X)/R)**2)
U_extrusion[U_extrusion < 0] = 0 # Vitesse nulle aux parois gauche/droite

# --- MODELE 2 : SYMETRIE CENTRALE ---
# La vitesse dépend du rayon r
r = np.sqrt(X**2 + Y**2)
U_sym = U_max * (1 - (r/R)**2)
U_sym[r > R] = 0 # Vitesse nulle si r > rayon (Cercle inscrit -> Coins à 0)

# --- TRACÉ ---
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1
c1 = axs[0].contourf(X, Y, U_extrusion, levels=20, cmap='viridis')
axs[0].set_title("Modèle 1 : Extrusion 2D\n(Surestimation)", fontsize=12, fontweight='bold')
axs[0].set_xlabel("x (mm)")
axs[0].set_ylabel("y (mm)")
axs[0].set_aspect('equal')
axs[0].add_patch(plt.Rectangle((-30,-30), 60, 60, fill=False, edgecolor='red', linewidth=2, label='Parois'))
# Annotation Coin
axs[0].text(20, 20, "Vitesse > 0\n(Erreur)", color='white', ha='center', fontsize=9, fontweight='bold')

# Plot 2
c2 = axs[1].contourf(X, Y, U_sym, levels=20, cmap='viridis')
axs[1].set_title("Modèle 2 : Symétrie Centrale\n(Correction des coins)", fontsize=12, fontweight='bold')
axs[1].set_xlabel("x (mm)")
axs[1].set_aspect('equal')
axs[1].add_patch(plt.Rectangle((-30,-30), 60, 60, fill=False, edgecolor='red', linewidth=2))
# Cercle inscrit
circle = plt.Circle((0, 0), 30, color='white', fill=False, linestyle='--', linewidth=1, alpha=0.7)
axs[1].add_patch(circle)
# Annotation Coin
axs[1].text(24, 24, "Zone Morte\n(V=0)", color='black', ha='center', fontsize=9, fontweight='bold')

# Barre de couleur commune
cbar = fig.colorbar(c2, ax=axs, orientation='right', fraction=0.05, pad=0.04)
cbar.set_label('Vitesse normalisée')

plt.suptitle("Comparaison des hypothèses de reconstruction du débit", fontsize=14)
plt.savefig("schema_symetrie.png", dpi=300, bbox_inches='tight')
plt.show()
