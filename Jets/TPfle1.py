import numpy as np
import matplotlib.pyplot as plt

rho = 1.14
g = 9.81

# =========================
# ÉTALONNAGE 2
# =========================
E_calib = np.array([1.38, 1.82, 1.88, 1.92, 1.96, 1.99, 2.02, 2.05, 2.07, 2.1,
                    2.13, 2.16, 2.20, 2.23, 2.25, 2.31, 2.36, 2.47, 2.54, 2.59])
Lmm = np.array([0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 1.8, 2.3,
                2.8, 3.3, 4.2, 5, 6, 8.3, 10.9, 19.8, 28.4, 36.3])

P = g * Lmm
U_calib = np.sqrt(2 * P / rho)

# Ajustement polynôme cubique
coeffs = np.polyfit(E_calib, U_calib, 3)
a, b, c, d = coeffs

def Um(E):
    return a*E**3 + b*E**2 + c*E + d

def dUm(E):
    return 3*a*E**2 + 2*b*E + c

# =========================
# x3 = 146.4 cm, Utot(h)
# =========================
h3 = np.array([400, 360, 320, 280, 240, 200, 160, 120, 80, 40, 0])
Erms3 = np.array([0.1, 0.12, 0.11, 90e-3, 80e-3, 80e-3, 0.1, 0.11, 0.11, 0.11, 80e-3])
Em3 = np.array([1.85, 2, 2.1, 2.2, 2.25, 2.2, 2.15, 2, 1.9, 1.75, 1.65])

Um3 = Um(Em3)
Uprimerms3 = dUm(Em3) * Erms3
Utot3 = Um3 + Uprimerms3

# =========================
# Profil à h = 220 mm, Utot(x)
# =========================
X = np.array([10, 25, 35, 50, 65, 80, 95, 110, 125, 140])
Erms_hfixe = np.array([1.4e-3, 10e-3, 20e-3, 50e-3, 80e-3, 85e-3, 90e-3, 90e-3, 90e-3, 90e-3])
Em_hfixe = np.array([2.6, 2.59, 2.59, 2.56, 2.52, 2.45, 2.39, 2.3, 2.25, 2.22])

Um_hfixe = Um(Em_hfixe)
Uprimerms_hfixe = dUm(Em_hfixe) * Erms_hfixe
Utot_hfixe = Um_hfixe + Uprimerms_hfixe



# Calcul des incertitudes verticales (sur U)
err_Utot = calculate_uncertainty(Em_hfixe, Erms_hfixe, dUm, sigma_calib3)

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

plt.xlabel("Position X")
plt.ylabel("Vitesse totale $U_{tot}$ (m/s)")
plt.title(r"Évolution de $U_{tot}(x)$ à hauteur fixe (avec incertitudes)")
plt.legend()
plt.grid(True)
plt.show()

# =========================
# Plot Utot(h)
# =========================
plt.figure(figsize=(8,6))
plt.plot(h3, Utot3, 'o-', label='x3 = 146.4 cm')
plt.gca().invert_yaxis()
plt.xlabel("U_tot (m/s)")
plt.ylabel("h (mm)")
plt.title("Profil vertical de U_tot en fonction de h")
plt.xlim(0, 25)
plt.ylim(0, 400)
plt.grid(True)
plt.legend()

# =========================
# Plot Utot(x)
# =========================
plt.figure(figsize=(8,6))
plt.plot(X, Utot_hfixe, 's-', color='darkorange', label='h = 220 mm')
plt.xlabel("x (cm)")
plt.ylabel("U_tot (m/s)")
plt.title("Profil longitudinal de U_tot à h = 220 mm")
plt.ylim(0, 25)
plt.grid(True)
plt.legend()

plt.show()
