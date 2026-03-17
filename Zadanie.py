import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Konfiguracja
years = ['1985', '1993', '2001', '2011']

# Zmienne na wyniki
results_eq = []
results_vari, results_gli, results_vigreen = [], [], []
bin_vari, bin_gli, bin_vigreen = [], [], []
perc_vari, perc_gli, perc_vigreen = [], [], []

# Wartości progów (dobrane empirycznie - można je lekko zmieniać, żeby wynik wyglądał lepiej)
thresh_vari = 0.05
thresh_gli = 0.05
thresh_vigreen = 0.05

# 2. Funkcje pomocnicze
def equalize_hsv(img):
    """Wyrównanie histogramu z wykorzystaniem konwersji na przestrzeń HSV"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2RGB)

def calc_indices(img):
    """Obliczanie wskaźników wegetacji"""
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    eps = 1e-8 
    
    vari = (g - r) / (g + r - b + eps)
    gli = (2 * g - r - b) / (2 * g + r + b + eps)
    vigreen = (g - r) / (g + r + eps)
    return vari, gli, vigreen

def plot_2x2(data, title, cmap=None, vmin=None, vmax=None):
    """Funkcja do rysowania siatki obrazów 2x2 z poprawionymi odstępami"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9)) 
    fig.suptitle(title, fontsize=16)
    
    for i, ax in enumerate(axes.flatten()):
        if cmap:
            im = ax.imshow(data[i], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(data[i])
        ax.set_title(f"{title.split()[0]} - {years[i]}")
        ax.axis('off')
        
    # Dodane odstępy: 
    # wspace - szerokość między kolumnami, hspace - wysokość między wierszami
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.85, top=0.9, wspace=0.3, hspace=0.3)
        
    if cmap and vmin is not None:
        # Przesunięcie colorbar
        cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7]) 
        fig.colorbar(im, cax=cbar_ax)
        
    plt.show()

# 3. Pętla główna analizująca zdjęcia
for year in years:
    
    path = f'images/{year}.png'
    img = cv2.imread(path)
    
    if img is None:
        print(f"❌ Nie znaleziono pliku: {path}")
        exit()
        
    # Wizualizacja (Wyrównanie HSV)
    results_eq.append(equalize_hsv(img))
    
    # Obliczenia indeksów na oryginale
    vari, gli, vigreen = calc_indices(img)
    results_vari.append(vari)
    results_gli.append(gli)
    results_vigreen.append(vigreen)
    
    # Progowanie (Obrazy binarne)
    b_vari = (vari > thresh_vari).astype(np.uint8)
    b_gli = (gli > thresh_gli).astype(np.uint8)
    b_vigreen = (vigreen > thresh_vigreen).astype(np.uint8)
    
    bin_vari.append(b_vari)
    bin_gli.append(b_gli)
    bin_vigreen.append(b_vigreen)
    
    #  Obliczanie procentu wylesiania
    perc_vari.append((np.sum(b_vari) / b_vari.size) * 100)
    perc_gli.append((np.sum(b_gli) / b_gli.size) * 100)
    perc_vigreen.append((np.sum(b_vigreen) / b_vigreen.size) * 100)

# 4. Wyświetlanie wyników 
plot_2x2(results_eq, "Zobrazowania RGB po equalizacji")

# Wyświetlanie map indeksów 
plot_2x2(results_vari, "VARI Index", cmap='jet', vmin=-0.2, vmax=0.2)
plot_2x2(results_gli, "GLI Index", cmap='jet', vmin=-0.2, vmax=0.2)
plot_2x2(results_vigreen, "VIGREEN Index", cmap='jet', vmin=-0.2, vmax=0.2)

# Wyświetlanie obrazów binarnych
plot_2x2(bin_vari, "VARI Thresholded", cmap='viridis')
plot_2x2(bin_gli, "GLI Thresholded", cmap='viridis')
plot_2x2(bin_vigreen, "VIGREEN Thresholded", cmap='viridis')

# 5. Rysowanie końcowego wykresu procentowego spadku
plt.figure(figsize=(8, 5))
plt.plot(years, perc_vari, marker='o', label='VARI')
plt.plot(years, perc_gli, marker='o', label='GLI')
plt.plot(years, perc_vigreen, marker='o', label='VIGREEN')

plt.ylim(0, 100)
plt.title("Percentage of Forest Area Over Time")
plt.xlabel("Year")
plt.ylabel("Percentage of Forest Area (%)")
plt.legend()
plt.grid(True)
plt.show()
