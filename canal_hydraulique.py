import numpy as np
import matplotlib.pyplot as plt

# --- 1. FONCTIONS MODELES (Physique) ---

def solve_depth_weir(Q, B, dz, g=9.81):
    """
    Résout l'équation de conservation de charge pour le seuil (déversoir large).
    Hypothèse : Ecoulement critique sur le seuil (contrôle hydraulique).
    """
    if Q <= 0: return np.nan
    # Charge critique sur le seuil (Ec = 3/2 * yc)
    yc = (Q**2 / (g * B**2))**(1/3)
    Ec = 1.5 * yc
    
    # Charge totale requise en amont = Ec + dz (hauteur de pelle)
    E_total_amont = Ec + dz
    
    # Résolution de Bernoulli amont : y1 + Q^2 / (2g B^2 y1^2) = E_total
    # Polynome : y1^3 - E_total * y1^2 + Q^2/(2gB^2) = 0
    A = Q**2 / (2 * g * B**2)
    coeffs = [1, -E_total_amont, 0, A]
    roots = np.roots(coeffs)
    
    # On garde la racine réelle > 0 correspondant au régime fluvial (la plus grande)
    real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    if not real_roots: return np.nan
    return max(real_roots)

def solve_depth_convergent(Q, B1, B2, g=9.81):
    """
    Résout l'équation pour le convergent.
    Hypothèse : Ecoulement critique au col (section B2).
    """
    if Q <= 0: return np.nan
    # Charge critique au col (section B2)
    yc2 = (Q**2 / (g * B2**2))**(1/3)
    Ec2 = 1.5 * yc2
    
    # Conservation de charge : E1 = E2 = Ec2 (négligeant pertes)
    E1 = Ec2
    
    # Résolution pour y1 (section B1)
    A = Q**2 / (2 * g * B1**2)
    coeffs = [1, -E1, 0, A]
    roots = np.roots(coeffs)
    
    real_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]
    if not real_roots: return np.nan
    return max(real_roots)

# --- 2. PARAMETRAGE ---

# Simulation Monte Carlo
N = 1000
Q_range_lmin = np.linspace(2, 60, 100) # L/min
Q_range = Q_range_lmin / 60000 # Conversion m3/s

# Incertitudes
u_Q_rel = 0.05 # 5% sur le débit
u_geo = 0.001  # 1 mm sur les dimensions géométriques

# Géométrie nominale
B_canal = 0.04   # Largeur canal
dz_seuil = 0.03  # Hauteur seuil
B1_conv = 0.048  # Largeur amont convergent
B2_conv = 0.018  # Largeur col convergent

# --- 3. DONNEES EXPERIMENTALES ---
# Seuil (Déversoir)
Qexp_seuil_lmin = np.array([49.15, 45.34, 40.45, 35.04, 29.94, 24.74, 20, 15, 10, 5])
# Transformation Y: 108mm est la référence zéro (fond) lue sur le limnimètre inversé
Yexp_seuil_raw = np.array([35, 36, 40, 43, 47, 50, 52, 57, 62, 67])
Yexp_seuil = np.abs(Yexp_seuil_raw - 108) / 1000 

# Convergent
Qexp_conv_lmin = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 48])
Yexp_conv_raw = np.array([88, 77, 69, 62, 55, 48, 43, 38, 32, 29])
Yexp_conv = np.abs(Yexp_conv_raw - 108) / 1000

# Incertitudes expérimentales
u_y_lecture = 0.002 # +/- 2mm (lecture visuelle + fluctuation surface)
u_Q_exp_seuil = Qexp_seuil_lmin * 0.05
u_Q_exp_conv = Qexp_conv_lmin * 0.05

# --- 4. CALCUL MONTE CARLO (Bandes de confiance) ---

res_seuil_mean, res_seuil_std = [], []
res_conv_mean, res_conv_std = [], []

for q in Q_range:
    # Génération des paramètres bruités
    q_mc = np.random.normal(q, q * u_Q_rel, N)
    B_mc = np.random.normal(B_canal, u_geo, N)
    dz_mc = np.random.normal(dz_seuil, u_geo, N)
    B1_mc = np.random.normal(B1_conv, u_geo, N)
    B2_mc = np.random.normal(B2_conv, u_geo, N)
    
    # Calcul Modèle Seuil
    y_seuil_mc = [solve_depth_weir(qi, bi, dzi) for qi, bi, dzi in zip(q_mc, B_mc, dz_mc)]
    res_seuil_mean.append(np.nanmean(y_seuil_mc))
    res_seuil_std.append(np.nanstd(y_seuil_mc))
    
    # Calcul Modèle Convergent
    y_conv_mc = [solve_depth_convergent(qi, b1i, b2i) for qi, b1i, b2i in zip(q_mc, B1_mc, B2_mc)]
    res_conv_mean.append(np.nanmean(y_conv_mc))
    res_conv_std.append(np.nanstd(y_conv_mc))

# Conversion numpy
res_seuil_mean = np.array(res_seuil_mean)
res_seuil_std = np.array(res_seuil_std)
res_conv_mean = np.array(res_conv_mean)
res_conv_std = np.array(res_conv_std)

# --- 5. GRAPHIQUES ---

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Graphique 1 : Seuil
ax1.errorbar(Qexp_seuil_lmin, Yexp_seuil, xerr=u_Q_exp_seuil, yerr=u_y_lecture, 
             fmt='k+', capsize=3, label='Mesures Exp.', zorder=5)
ax1.plot(Q_range_lmin, res_seuil_mean, 'b-', label='Modèle Théorique (Bernoulli)')
ax1.fill_between(Q_range_lmin, res_seuil_mean - 2*res_seuil_std, res_seuil_mean + 2*res_seuil_std, 
                 color='blue', alpha=0.2, label='Incertitude Modèle (95%)')

ax1.set_title("Ligne d'eau amont - Seuil (Déversoir)")
ax1.set_xlabel("Débit $Q$ (L/min)")
ax1.set_ylabel("Profondeur amont $y_1$ (m)")
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Graphique 2 : Convergent
ax2.errorbar(Qexp_conv_lmin, Yexp_conv, xerr=u_Q_exp_conv, yerr=u_y_lecture, 
             fmt='k+', capsize=3, label='Mesures Exp.', zorder=5)
ax2.plot(Q_range_lmin, res_conv_mean, 'r-', label='Modèle Théorique (Bernoulli)')
ax2.fill_between(Q_range_lmin, res_conv_mean - 2*res_conv_std, res_conv_mean + 2*res_conv_std, 
                 color='red', alpha=0.2, label='Incertitude Modèle (95%)')

ax2.set_title("Ligne d'eau amont - Convergent")
ax2.set_xlabel("Débit $Q$ (L/min)")
ax2.set_ylabel("Profondeur amont $y_1$ (m)")
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.savefig('resultats_canal_hydraulique.png')
plt.show()