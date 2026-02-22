import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

data = pd.read_csv("data/data2.csv", header=None)

scaler = StandardScaler()
normalized_features = scaler.fit_transform(data)

wcss = []
iterations = []
k_range = range(2, 11)



for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=np.random.randint(0, 100), max_iter=150)
    kmeans.fit(normalized_features)
    wcss.append(kmeans.inertia_)
    iterations.append(kmeans)

plt.figure(figsize=(8, 5))
plt.plot(k_range, wcss, marker='o', linestyle='-', color='b')
plt.title("WCSS w zależności od liczby klastrów")
plt.xlabel("Liczba klastrów (k)")
plt.ylabel("WCSS")



kmeans_3 = KMeans(n_clusters=3)
kmeans_3.fit(normalized_features)
labels = kmeans_3.labels_
centroidy = kmeans_3.cluster_centers_

kolor = [plt.cm.viridis(label / 3) for label in labels]
kolorCentroida = [plt.cm.viridis(i / 3) for i in range(3)]

zmienna_od_zaleznosci = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
nazwy = [nazwa + " (cm)" for nazwa in ["Długość działki kielicha", "Szerokość działki kielicha", "Długość płatka", "Szerokość płatka"]]

fig, axes = plt.subplots(3, 2, figsize=(8,11))

for i, (lewastrona, prawastrona) in enumerate(zmienna_od_zaleznosci):
    ax = axes[i // 2, i % 2]
    ax.scatter(normalized_features[:, lewastrona], normalized_features[:, prawastrona], c=kolor,s=30,alpha=0.3,label='dane')
    ax.scatter(centroidy[:, lewastrona], centroidy[:, prawastrona], c=kolorCentroida, marker='*',s=200,label='Centroidy')
    ax.set_xlabel(nazwy[lewastrona])
    ax.set_ylabel(nazwy[prawastrona])
    ax.legend(loc='best')


plt.tight_layout()
plt.show()

print("Liczba klastrów | Liczba iteracji | WCSS")
print("-" * 40)
for k, w, iters in zip(k_range, wcss, [kmeans.n_iter_ for kmeans in iterations]):
    print(f"{k:<15} | {iters:<16} | {w:.2f}")


