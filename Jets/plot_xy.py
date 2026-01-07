import numpy as np
import matplotlib.pyplot as plt
import re
import os

# --- FONCTION DE LECTURE ROBUSTE ---
def read_fluent_xy(filename):
    """Lit un fichier .xy Fluent et extrait les données."""
    x_data = []
    y_data = []
    curve_name = filename 
    
    # Vérification de l'existence du fichier
    if not os.path.exists(filename):
        # On essaye avec l'extension .xy ou .txt si l'utilisateur a oublié
        if os.path.exists(filename + ".xy"):
            filename += ".xy"
        elif os.path.exists(filename + ".txt"):
            filename += ".txt"
        else:
            print(f"❌ ERREUR CRITIQUE : Le fichier '{filename}' est introuvable.")
            return np.array([]), np.array([]), ""

    print(f"✅ Lecture du fichier : {filename}")
    
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Nom de la courbe
            if "xy/key/label" in line:
                match = re.search(r'"(.*?)"', line)
                if match:
                    curve_name = match.group(1)
                continue
            
            # Ignorer métadonnées
            if line.startswith('('):
                continue
            
            # Données
            try:
                parts = line.split()
                if len(parts) == 2:
                    x_data.append(float(parts[0]))
                    y_data.append(float(parts[1]))
            except ValueError:
                continue
                
        return np.array(x_data), np.array(y_data), curve_name
        
    except Exception as e:
        print(f"Erreur lors de la lecture : {e}")
        return np.array([]), np.array([]), ""

# ==========================================
# 1. DIAGNOSTIC DU DOSSIER COURANT
# ==========================================
cwd = os.getcwd()
print(f"\n📂 Le script s'exécute dans : {cwd}")
print("📄 Liste des fichiers trouvés ici :")
files_in_dir = os.listdir(cwd)
print(files_in_dir)
print("-" * 30)

# Vérifiez si vos fichiers sont dans cette liste ! 
# S'ils n'y sont pas, déplacez-les dans le dossier affiché ci-dessus.

# ==========================================
# 2. CHARGEMENT ET TRACÉ
# ==========================================

# Mettez ici le nom EXACT tel qu'il apparaît dans la liste ci-dessus
# Si votre fichier s'appelle "vit_long.txt", écrivez "vit_long.txt"
file_long = 'vit_long'   
file_trans = 'vit_trans' 

x_long, u_long, label_long = read_fluent_xy(file_long)
x_trans, u_trans, label_trans = read_fluent_xy(file_trans)

# Si aucun fichier n'est trouvé, on arrête
if len(x_long) == 0 and len(x_trans) == 0:
    print("\n⚠️ AUCUNE DONNÉE TROUVÉE. Vérifiez l'emplacement des fichiers.")
else:
    # Tracé
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Graphique 1
    if len(x_long) > 0:
        idx = np.argsort(x_long)
        ax1.plot(x_long[idx], u_long[idx], 'b-', linewidth=2, label=label_long)
        ax1.set_xlabel('Position Axiale (m)')
        ax1.set_ylabel('Vitesse (m/s)')
        ax1.set_title('Profil Longitudinal')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()

    # Graphique 2
    if len(x_trans) > 0:
        idx = np.argsort(x_trans)
        ax2.plot(x_trans[idx], u_trans[idx], 'r-', linewidth=2, label=label_trans)
        ax2.set_xlabel('Position Radiale (m)')
        ax2.set_ylabel('Vitesse (m/s)')
        ax2.set_title('Profil Transversal')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

    plt.tight_layout()
    plt.show()