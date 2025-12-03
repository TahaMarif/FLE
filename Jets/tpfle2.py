import numpy as np
import matplotlib.pyplot as plt

rho = 1.14
g = 9.81

# --- Fonctions pour l'incertitude ---
def get_calibration_error(E_calib, U_calib, func_U_from_E):
    """Calcule l'erreur type (RMSE) de la calibration."""
    U_model = func_U_from_E(np.array(E_calib))
    residuals = np.array(U_calib) - U_model
    return np.std(residuals)

def calculate_uncertainty(Em_vals, Erms_vals, func_dU_dE, sigma_calib, delta_Em=0.01, delta_Erms=0.002):
    """Calcule l'incertitude totale sur Utot (m/s)."""
    Em = np.array(Em_vals)
    sensitivity = np.abs(func_dU_dE(Em))
    
    # 1. Incertitude sur la vitesse moyenne Um
    u_Um = np.sqrt(sigma_calib**2 + (sensitivity * delta_Em)**2)
    
    # 2. Incertitude sur la fluctuation U'rms
    u_Uprime = sensitivity * delta_Erms
    
    # 3. Incertitude totale
    return np.sqrt(u_Um**2 + u_Uprime**2)

# ==========================================
# 1. CALIBRATIONS
# ==========================================

# --- Calib 1 (X1, X2) ---
Lmm1 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1, 1.4, 1.6, 3, 4.1, 6.5, 9.3, 24]
P1 = [g * e for e in Lmm1]
U1 = [(2 * e / rho) ** 0.5 for e in P1]
E1 = [1.38, 1.82, 1.86, 1.92, 1.96, 1.97, 2.01, 2.06, 2.08, 2.11, 2.13, 2.22, 2.27, 2.35, 2.41, 2.6]
a1, b1, c1, d1 = np.polyfit(E1, U1, 3)
def U_from_E1(E): return a1*E**3 + b1*E**2 + c1*E + d1
def dU_dE1(E): return 3*a1*E**2 + 2*b1*E + c1
sigma_calib1 = get_calibration_error(E1, U1, U_from_E1)

# --- Calib 2 (X3) ---
E_calib3 = np.array([1.38, 1.82, 1.88, 1.92, 1.96, 1.99, 2.02, 2.05, 2.07, 2.1, 2.13, 2.16, 2.20, 2.23, 2.25, 2.31, 2.36, 2.47, 2.54, 2.59])
Lmm3 = np.array([0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.3, 2.8, 3.3, 4.2, 5, 6, 8.3, 10.9, 19.8, 28.4, 36.3])
P3 = g * Lmm3
U_calib3 = np.sqrt(2 * P3 / rho)
a3, b3, c3, d3 = np.polyfit(E_calib3, U_calib3, 3)
def U_from_E3(E): return a3*E**3 + b3*E**2 + c3*E + d3
def dU_dE3(E): return 3*a3*E**2 + 2*b3*E + c3
sigma_calib3 = get_calibration_error(E_calib3, U_calib3, U_from_E3)

# ==========================================
# 2. TRAITEMENT DES DONNÉES
# ==========================================
# --- X1 ---
h1 = np.array([330, 268, 260, 250, 240, 230, 220, 210, 200, 190, 165, 130])
Erms1 = np.array([12, 10, 20, 1.7, 1.7, 1.6, 1.4, 1.4, 1.6, 2.2, 7, 15]) * 1e-3
Em1 = np.array([1.5, 1.7, 2.68, 2.68, 2.68, 2.68, 2.68, 2.68, 2.68, 2.69, 1.7, 1.6])
Utot1 = U_from_E1(Em1) + dU_dE1(Em1) * Erms1
err1 = calculate_uncertainty(Em1, Erms1, dU_dE1, sigma_calib1)

# --- X2 ---
h2 = np.array([130, 150, 156, 167, 173, 178, 183, 190, 195, 200, 205, 210, 215, 220, 230, 240, 250, 260, 270, 280, 290, 310, 330])
Erms2 = np.array([60e-3, 0.13, 0.15, 0.15, 0.12, 0.13, 0.11, 90e-3, 72e-3, 50e-3, 40e-3, 30e-3, 25e-3, 21e-3, 22e-3, 35e-3, 65e-3, 0.1, 0.12, 0.15, 0.14, 70e-3, 40e-3])
Em2 = np.array([1.66, 1.8, 1.89, 2.1, 2.3, 2.35, 2.44, 2.53, 2.6, 2.63, 2.66, 2.68, 2.68, 2.68, 2.68, 2.66, 2.6, 2.5, 2.3, 2.1, 1.9, 1.7, 1.6])
Utot2 = U_from_E1(Em2) + dU_dE1(Em2) * Erms2
err2 = calculate_uncertainty(Em2, Erms2, dU_dE1, sigma_calib1)

# --- X3 ---
h3 = np.array([400, 360, 320, 280, 240, 200, 160, 120, 80, 40, 0])
Erms3 = np.array([0.1, 0.12, 0.11, 90e-3, 80e-3, 80e-3, 0.1, 0.11, 0.11, 0.11, 80e-3])
Em3 = np.array([1.85, 2, 2.1, 2.2, 2.25, 2.2, 2.15, 2, 1.9, 1.75, 1.65])
Utot3 = U_from_E3(Em3) + dU_dE3(Em3) * Erms3
err3 = calculate_uncertainty(Em3, Erms3, dU_dE3, sigma_calib3)


# --- 3. TRAITEMENT DU JEU DE DONNÉES AXIAL ---
X = np.array([10, 25, 35, 50, 65, 80, 95, 110, 125, 140]) # Position
Erms_hfixe = np.array([1.4e-3, 10e-3, 20e-3, 50e-3, 80e-3, 85e-3, 90e-3, 90e-3, 90e-3, 90e-3])
Em_hfixe = np.array([2.6, 2.59, 2.59, 2.56, 2.52, 2.45, 2.39, 2.3, 2.25, 2.22])

# Calcul des vitesses
Um_val = U_from_E3(Em_hfixe)
Uprime_val = dU_dE3(Em_hfixe) * Erms_hfixe
Utot_hfixe = Um_val + Uprime_val

# Calcul des incertitudes verticales (sur U)
err_Utot = calculate_uncertainty(Em_hfixe, Erms_hfixe, dU_dE3, sigma_calib3)

# --- 4. TRACÉ ---
delta_x = 1.0 # Incertitude sur la position X en mm (ou cm selon l'unité de X)

plt.figure(figsize=(9, 6))

plt.errorbar(X, Utot_hfixe, 
             xerr=delta_x, 
             yerr=err_Utot, 
             fmt='-o', 
             color='purple',
             ecolor='gray',
             capsize=3, 
             label='Profil à h = 220 mm')

plt.xlabel("Position $x$ (cm)")
plt.ylabel("Vitesse totale $U_{tot}$ (m/s)")
plt.title(r"Évolution de $U_{tot}(x)$ à hauteur fixe avec incertitudes")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# 3. TRACÉ AVEC DOUBLE INCERTITUDE
# ==========================================
delta_h = 1.0 # mm

plt.figure(figsize=(10, 7))

# On ajoute xerr=delta_h pour avoir les barres horizontales
plt.errorbar(h1, Utot1, xerr=delta_h, yerr=err1, fmt='o-', label='X1 = 9.2 cm', capsize=3)
plt.errorbar(h2, Utot2, xerr=delta_h, yerr=err2, fmt='s-', label='X2 = 36.2 cm', capsize=3)
plt.errorbar(h3, Utot3, xerr=delta_h, yerr=err3, fmt='x-', label='X3 = 146.4 cm', capsize=3)

plt.xlabel("Position $h$ (mm)")
plt.ylabel("Vitesse totale $U_{tot}(h)$ (m/s)")
plt.title(f"Profils de vitesse avec incertitudes associées")
plt.legend()
plt.grid(True)
plt.show()