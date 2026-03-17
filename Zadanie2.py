import rasterio
import numpy as np
import matplotlib.pyplot as plt

# 1. Konfiguracja 
years = ['1985', '1993', '2001', '2011']
 
multi_files = ['1985api.tif', '1993api.tif', '2001api.tif', '2011api.tif']

ndvi_maps = []
ndvi_binary = []
forest_percentages = []

# Próg NDVI
threshold_ndvi = 0.4 

print("--- Rozpoczynam analizę wielospektralną ---")

for file in multi_files:
    path = f'images_geotiff/{file}'
    try:
        with rasterio.open(path) as src:
            # Wczytywanie kanałów: Red (3) i NIR (4) [cite: 24]
            red = src.read(3).astype('float32')
            nir = src.read(4).astype('float32')
            
            # Obliczanie NDVI [cite: 24, 33]
            # Wzór: (NIR - RED) / (NIR + RED)
            denom = nir + red
            ndvi = np.divide((nir - red), denom, out=np.zeros_like(nir), where=denom!=0)
            ndvi_maps.append(ndvi)
            
            # Generowanie obrazu binarnego (klasyfikacja) [cite: 28]
            binary = (ndvi > threshold_ndvi).astype(np.uint8)
            ndvi_binary.append(binary)
            
            # Obliczanie procentu zalesienia [cite: 29]
            perc = (np.sum(binary) / binary.size) * 100
            forest_percentages.append(perc)
            print(f"Przetworzono {file}: {perc:.2f}% powierzchni roślinnej")

    except Exception as e:
        print(f"❌ Błąd przy pliku {file}: {e}")

# 2. Wizualizacja map NDVI [cite: 27, 30]
def plot_ndvi_results(data, title, cmap='RdYlGn'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        im = ax.imshow(data[i], cmap=cmap, vmin=-1, vmax=1)
        ax.set_title(f"NDVI - {years[i]}")
        ax.axis('off')
    
    plt.subplots_adjust(wspace=0.2, hspace=0.2, right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()

# 3. Wizualizacja obrazów binarnych [cite: 28, 30]
def plot_binary_results(data, title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(data[i], cmap='viridis')
        ax.set_title(f"Zasięg roślinności - {years[i]}")
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Wywołanie okien z mapami
plot_ndvi_results(ndvi_maps, "Mapy wskaźnika NDVI")
plot_binary_results(ndvi_binary, "Obrazy binarne (Roślinność vs Brak)")

# 4. Wykres słupkowy zmian zalesienia [cite: 29, 30]
plt.figure(figsize=(10, 6))
bars = plt.bar(years, forest_percentages, color='forestgreen', edgecolor='black')
plt.title("Analiza wylesiania na podstawie wskaźnika NDVI", fontsize=14)
plt.xlabel("Rok", fontsize=12)
plt.ylabel("Procent powierzchni leśnej (%)", fontsize=12)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Dodanie wartości nad słupkami
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.show()
