# Para usar el programa, debe abrir el codigo y ejecutarlo en la misma carpeta donde este la imagen.
# La imagen la puede editar en paint y poner cualquier figura que usted quiera siempre y cuando el fondo sea blanco y la figura negra.
# En el codigo se puede cambiar valores de corriente y ajustar las dimensiones de la imagen, reocmiendo leer el codigo.

import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

# Cargar la imagen
img = cv2.imread('Prueba.jpg', cv2.IMREAD_GRAYSCALE)

# Obtener las dimensiones de la imagen
img_height, img_width = img.shape
real_espira_size = 0.115  # Cuanto mide el hancho de la imagen en metros
scale_factor = real_espira_size / img_width  # Factor de escala en metros por píxel

# Preprocesamiento: umbral para obtener una imagen binaria
_, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

# Encontrar los contornos de la curva
contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extraer coordenadas de los contornos y aplicar el factor de escala
x_values = []
y_values = []

for contour in contours:
    for point in contour:
        # Escalar las coordenadas por el factor de escala
        x_values.append(point[0][0] * scale_factor)  # Coordenada X
        y_values.append(point[0][1] * scale_factor)  # Coordenada Y

# Definir la función que calcula el campo magnetico por medio de la "Ley de Biot-Savart".
def MagneticField(x, y, wire, I=1):
    mu0 = 4 * np.pi * 10**(-7)
    c = mu0 * I / (4 * np.pi)
    xA, yA = wire[0][0], wire[0][1]
    xB, yB = wire[1][0], wire[1][1]
    r1 = np.sqrt((x - xA)**2 + (y - yA)**2)
    r2 = np.sqrt((x - xB)**2 + (y - yB)**2)
    L = np.sqrt((xB - xA)**2 + (yB - yA)**2)
    CosTheta1 = (r2**2 - r1**2 - L**2) / (2 * L * r1)
    CosTheta2 = (r2**2 - r1**2 + L**2) / (2 * L * r2)
    distance = np.sqrt(2 * r1**2 * r2**2 + 2 * r1**2 * L**2 + 2 * r2**2 * L**2 - r1**4 - r2**4 - L**4) / (2 * L)
    Bfield = c * (CosTheta2 - CosTheta1) / distance
    return Bfield

# Generar el plano (x,y).
Calidad = 100   #  pixeles x pixeles
X_lim = 0.07    #  Limites de X
Y_lim = 0.07    #  Limiste de Y

x = np.linspace(-X_lim, X_lim, Calidad)
y = np.linspace(-Y_lim, Y_lim, Calidad)
[x, y] = np.meshgrid(x, y)

sum_Bs = np.zeros_like(x)

# Iterar sobre los contornos y calcular el campo magnético solo entre puntos vecinos
for contour in contours:
    for i in range(len(contour) - 1):
        # Tomar puntos consecutivos del contorno
        wire = [(contour[i][0][0] * scale_factor - 0.0585 , contour[i][0][1] * scale_factor - 0.058 ),
                (contour[i+1][0][0] * scale_factor - 0.0585 , contour[i+1][0][1] * scale_factor - 0.058 )]
        B = MagneticField(x, y, wire)
        sum_Bs += B  # Sumar el campo magnético calculado para este par de puntos
        
        progress = i + 1
        percentage = (progress / (len(contour) - 1)) * 100
        print(f"Progreso: {progress}/{(len(contour)-1)} ({percentage:.2f}%)", end='\r')

# Visualización del campo magnético
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
im = ax.pcolormesh(x, y, np.log10(np.abs(sum_Bs)), vmin=-5, vmax=-3, cmap="jet")
plt.title("Campo Magnético en un Trébol", fontweight='bold') 
ax.set_xlim(-X_lim, X_lim)
ax.set_xlabel(r"$x$ [m]")
ax.set_ylim(-Y_lim, Y_lim)
ax.set_ylabel(r"$y$ [m]")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.01)
cb = fig.colorbar(im, ax=ax, orientation="vertical", cax=cax)
cb.set_label("$\log(B)$", labelpad=5)

plt.show()

