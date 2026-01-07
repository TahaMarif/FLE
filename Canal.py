import numpy as np
import matplotlib.pyplot as plt

def plot_equation_7_range_55():
    # 1. Constantes physiques (Canal Hydraulique-2)
    g = 9.81
    B = 0.04          # Largeur 40mm 
    delta_h = 0.03    # Hauteur du seuil standard 30mm 
    
    # 2. Configuration des plages de données
    # Axe X : Débit Q de 0 à 55 L/min
    Q_L_min = np.linspace(0.1, 55, 1000) 
    
    # Conversion en unités SI (m^3/s) pour les calculs
    # 1 L/min = 1 / 60000 m^3/s
    Q_si = Q_L_min / 60000.0
    
    # Axe Y : Hauteur amont h1 (en m)
    # On cherche h1 entre la hauteur du seuil (0.03m) et une hauteur max raisonnable (0.15m)
    h1_vals = np.linspace(delta_h, 0.15, 1000)
    
    # 3. Création de la grille (Meshgrid)
    Q_grid_si, H1_grid = np.meshgrid(Q_si, h1_vals)
    
    # Calcul du débit linéique q = Q / B
    q_grid = Q_grid_si / B
    
    # 4. Équation 7 [cite: 309]
    # h1 + q^2 / (2g * h1^2) = 1.5 * (q^2 / g)^(1/3) + delta_h
    
    # Énergie spécifique en amont (Terme de gauche)
    E_amont = H1_grid + (q_grid**2) / (2 * g * H1_grid**2)
    
    # Énergie critique sur le seuil + Delta h (Terme de droite)
    E_critique_seuil = 1.5 * np.cbrt((q_grid**2) / g) + delta_h
    
    # On cherche la solution Z = 0
    Z = E_amont - E_critique_seuil

    # 5. Tracé du graphique
    plt.figure(figsize=(10, 6))
    
    # On utilise contour pour tracer la ligne exacte où l'équation est satisfaite
    # X = Débit en L/min, Y = Hauteur en mm
    CS = plt.contour(Q_L_min, h1_vals * 1000, Z, levels=[0], colors='blue', linewidths=2)
    
    # Légende pour le contour
    if CS.collections:
        CS.collections[0].set_label(r'Hauteur théorique $h_1$ (Régime Fluvial)')

    # Ajout de la hauteur physique du seuil pour référence
    plt.axhline(y=delta_h * 1000, color='red', linestyle='--', label=f'Hauteur du seuil ({delta_h*1000}mm)')

    # Mise en forme
    plt.title(r'Relation $h_1$ vs $Q$ (Équation 7) pour $0 < Q < 55$ L/min')
    plt.xlabel('Débit Q (L/min)')
    plt.ylabel('Hauteur d\'eau amont $h_1$ (mm)')
    plt.xlim(0, 55)
    plt.ylim(0, 100) # On zoome sur 0-100mm car le canal est peu profond
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.show()

if __name__ == "__main__":
    plot_equation_7_range_55()