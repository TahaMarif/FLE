import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 0. PARAMÈTRES PHYSIQUES ET INSTRUMENTAUX
# ==========================================
rho = 1.14
g = 9.81

# --- Spécifications Constructeurs (Incertitudes) ---
# 1. Carte NI 9215
NI_GAIN = 0.0002        # 0.02%
NI_RANGE = 10.4         # 10.4 V
NI_OFFSET_ERR = 0.00014 
NI_CONST_ERR = NI_RANGE * NI_OFFSET_ERR # ≈ 1.456 mV

# 2. Pitot FCO318
PITOT_REL_ERR = 0.0025 # 0.25% sur la Pression

# --- Fonctions de calcul d'incertitude ---

def get_voltage_uncertainty_NI(V_val):
    """Calcule l'incertitude absolue sur une tension (NI 9215)."""
    # Formule : Gain * Lecture + Gamme * Offset
    return np.abs(V_val) * NI_GAIN + NI_CONST_ERR

def get_pitot_uncertainty_velocity(U_val):
    """Calcule l'incertitude sur la vitesse de référence (Pitot)."""
    # dU/U = 0.5 * dP/P = 0.5 * 0.25% = 0.125%
    return U_val * (0.5 * PITOT_REL_ERR)

def get_calibration_error(E_calib, U_calib, func_U_from_E):
    """
    Calcule l'incertitude globale du modèle d'étalonnage (Sigma_modèle).
    Combine RMSE (résidus) + Incertitude systématique du Pitot.
    """
    # 1. Erreur statistique du modèle (RMSE)
    U_model = func_U_from_E(np.array(E_calib))
    residuals = np.array(U_calib) - U_model
    rmse = np.std(residuals)
    
    # 2. Erreur systématique moyenne du Pitot sur la plage
    u_pitot_mean = np.mean(get_pitot_uncertainty_velocity(np.array(U_calib)))
    
    # Combinaison quadratique
    return np.sqrt(rmse**2 + u_pitot_mean**2)

def calculate_uncertainty(Em_vals, Erms_vals, func_dU_dE, sigma_calib):
    """
    Calcule l'incertitude totale sur Utot (m/s).
    Prend en compte : Modèle + Pitot + Carte NI (Moyenne & RMS)
    """
    Em = np.array(Em_vals)
    Erms = np.array(Erms_vals)
    sensitivity = np.abs(func_dU_dE(Em))
    
    # 1. Incertitude sur les tensions (Instrument)
    u_Em_volt = get_voltage_uncertainty_NI(Em)
    u_Erms_volt = get_voltage_uncertainty_NI(Erms)
    
    # 2. Propagation sur la vitesse moyenne Um
    # On combine l'erreur du modèle (sigma_calib) et l'erreur de lecture (sensibilité * u_volt)
    u_Um = np.sqrt(sigma_calib**2 + (sensitivity * u_Em_volt)**2)
    
    # 3. Propagation sur la fluctuation U'rms
    u_Uprime = sensitivity * u_Erms_volt
    
    # 4. Incertitude totale (somme quadratique)
    return np.sqrt(u_Um**2 + u_Uprime**2)

# ==========================================
# 1. CALIBRATIONS (FIT & ERREURS)
# ==========================================

# --- Calib 1 (Pour X1, X2) ---
Lmm1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.4, 1.6, 3, 4.1, 6.5, 9.3, 24]
P1 = [g * e for e in Lmm1]
U1 = [(2 * e / rho) ** 0.5 for e in P1]
E1 = [1.38, 1.82, 1.86, 1.92, 1.96, 1.97, 2.01, 2.06, 2.08, 2.11, 2.13, 2.22, 2.27, 2.35, 2.41, 2.6]

a1, b1, c1, d1 = np.polyfit(E1, U1, 3)
def U_from_E1(E): return a1*E**3 + b1*E**2 + c1*E + d1
def dU_dE1(E): return 3*a1*E**2 + 2*b1*E + c1

# Calcul de l'incertitude modèle 1
sigma_calib1 = get_calibration_error(E1, U1, U_from_E1)
print(f"Sigma Global Calib 1 (Modèle + Pitot) : {sigma_calib1:.3f} m/s")

# --- Calib 2 (Pour X3 et Profil Axial) ---
E_calib3 = np.array([1.38, 1.82, 1.88, 1.92, 1.96, 1.99, 2.02, 2.05, 2.07, 2.1, 2.13, 2.16, 2.20, 2.23, 2.25, 2.31, 2.36, 2.47, 2.54, 2.59])
Lmm3 = np.array([0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.3, 2.8, 3.3, 4.2, 5, 6, 8.3, 10.9, 19.8, 28.4, 36.3])
P3 = g * Lmm3
U_calib3 = np.sqrt(2 * P3 / rho)

a3, b3, c3, d3 = np.polyfit(E_calib3, U_calib3, 3)
def U_from_E3(E): return a3*E**3 + b3*E**2 + c3*E + d3
def dU_dE3(E): return 3*a3*E**2 + 2*b3*E + c3

# Calcul de l'incertitude modèle 3
sigma_calib3 = get_calibration_error(E_calib3, U_calib3, U_from_E3)
print(f"Sigma Global Calib 3 (Modèle + Pitot) : {sigma_calib3:.3f} m/s")


# ==========================================
# 2. TRAITEMENT DES PROFILS (h)
# ==========================================

# --- Données X1 ---
h1 = np.array([330, 268, 260, 250, 240, 230, 220, 210, 200, 190, 165, 130])
Erms1 = np.array([12, 10, 20, 1.7, 1.7, 1.6, 1.4, 1.4, 1.6, 2.2, 7, 15]) * 1e-3
Em1 = np.array([1.5, 1.7, 2.68, 2.68, 2.68, 2.68, 2.68, 2.68, 2.68, 2.69, 1.7, 1.6])

Utot1 = U_from_E1(Em1) + dU_dE1(Em1) * Erms1
err1 = calculate_uncertainty(Em1, Erms1, dU_dE1, sigma_calib1)

# --- Données X2 ---
h2 = np.array([130, 150, 156, 167, 173, 178, 183, 190, 195, 200, 205, 210, 215, 220, 230, 240, 250, 260, 270, 280, 290, 310, 330])
Erms2 = np.array([60e-3, 0.13, 0.15, 0.15, 0.12, 0.13, 0.11, 90e-3, 72e-3, 50e-3, 40e-3, 30e-3, 25e-3, 21e-3, 22e-3, 35e-3, 65e-3, 0.1, 0.12, 0.15, 0.14, 70e-3, 40e-3])
Em2 = np.array([1.66, 1.8, 1.89, 2.1, 2.3, 2.35, 2.44, 2.53, 2.6, 2.63, 2.66, 2.68, 2.68, 2.68, 2.68, 2.66, 2.6, 2.5, 2.3, 2.1, 1.9, 1.7, 1.6])

Utot2 = U_from_E1(Em2) + dU_dE1(Em2) * Erms2
err2 = calculate_uncertainty(Em2, Erms2, dU_dE1, sigma_calib1)

# --- Données X3 ---
h3 = np.array([400, 360, 320, 280, 240, 200, 160, 120, 80, 40, 0])
Erms3 = np.array([0.1, 0.12, 0.11, 90e-3, 80e-3, 80e-3, 0.1, 0.11, 0.11, 0.11, 80e-3])
Em3 = np.array([1.85, 2, 2.1, 2.2, 2.25, 2.2, 2.15, 2, 1.9, 1.75, 1.65])

Utot3 = U_from_E3(Em3) + dU_dE3(Em3) * Erms3
err3 = calculate_uncertainty(Em3, Erms3, dU_dE3, sigma_calib3)


# ==========================================
# 3. TRAITEMENT PROFIL AXIAL (x)
# ==========================================
X_ax = np.array([10, 25, 35, 50, 65, 80, 95, 110, 125, 140]) # Position
Erms_hfixe = np.array([1.4e-3, 10e-3, 20e-3, 50e-3, 80e-3, 85e-3, 90e-3, 90e-3, 90e-3, 90e-3])
Em_hfixe = np.array([2.6, 2.59, 2.59, 2.56, 2.52, 2.45, 2.39, 2.3, 2.25, 2.22])

# Calcul (Utilise la Calib 3 car les valeurs montent à 2.6V, cohérent avec calib3)
Utot_hfixe = U_from_E3(Em_hfixe) + dU_dE3(Em_hfixe) * Erms_hfixe
err_Utot_ax = calculate_uncertainty(Em_hfixe, Erms_hfixe, dU_dE3, sigma_calib3)


# ==========================================
# 4. TRACÉS
# ==========================================

# --- Figure 1 : Profil Axial ---
delta_x = 1.0 # Incertitude position (mm)

plt.figure(figsize=(9, 6))
plt.errorbar(X_ax, Utot_hfixe, 
             xerr=delta_x, 
             yerr=err_Utot_ax, 
             fmt='-o', 
             color='purple',
             ecolor='gray',
             capsize=3, 
             label='Profil Axial (h=220mm)')

plt.xlabel("Position $x$ (mm)") # Vérifiez si vos données X sont en cm ou mm
plt.ylabel("Vitesse totale $U_{tot}$ (m/s)")
plt.title(r"Évolution de $U_{tot}(x)$ le long de l'axe (avec incertitudes)")
plt.legend()
plt.grid(True)
plt.show()

# --- Figure 2 : Profils Transversaux ---
delta_h = 1.0 # Incertitude position (mm)

plt.figure(figsize=(10, 7))
plt.errorbar(h1, Utot1, xerr=delta_h, yerr=err1, fmt='o-', label='X1 = 9.2 cm', capsize=3)
plt.errorbar(h2, Utot2, xerr=delta_h, yerr=err2, fmt='s-', label='X2 = 36.2 cm', capsize=3)
plt.errorbar(h3, Utot3, xerr=delta_h, yerr=err3, fmt='x-', label='X3 = 146.4 cm', capsize=3)

plt.xlabel("Position transversale $h$ (mm)")
plt.ylabel("Vitesse totale $U_{tot}$ (m/s)")
plt.title(f"Profils transversaux de vitesse (Incertitudes NI 9215 + Pitot)")
plt.legend()
plt.grid(True)
plt.show()