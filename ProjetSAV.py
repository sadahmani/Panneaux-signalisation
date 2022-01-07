#Importation des librairies nécessaire
import cv2 as cv
import pandas as pd
import numpy as np
from skimage.feature import hog 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import itertools
import cmath
import os


#declaration des fonctions 
## fonction pour le chargement des images vers un vecteur 
def Load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
            images.append(img)
    return images

#fonction pour la conversion des images vers l'espace Ycbcr
def Conversion_to_YCbCr(listimage) :
    channels = [] 
    y = [] 
    Cb = [] 
    Cr = [] 
    Images_Convertis = []
    for i in range(len(listimage)) :
        channels.append(cv.split(listimage[i])) 
        y.append(0.299*channels[i][0] + 0.587*channels[i][1] + 0.114*channels[i][2])
        Cb.append(-0.1687*channels[i][0] - 0.3313*channels[i][1] + 0.5*channels[i][2] + 128)
        Cr.append(0.5*channels[i][0] - 0.4187*channels[i][1] - 0.0813*channels[i][2] + 128)
        Images_Convertis.append(cv.merge([y[i],Cb[i],Cr[i]]))
        Images_Convertis[i] = np.uint8(Images_Convertis[i])
    return Images_Convertis

#fonction pour le redemnsionmenet des images 
def resize_images(listimage) :
    dim = (64, 128)
    for i in range(len(listimage)) :
        listimage[i] = cv.resize(listimage[i],dim)

#fonction pour le calcule de HOG 
def Calcul_hog(listimage) :
    fv = []
    hog_image = []
    for i in range(len(listimage)) :
        a, b = hog(listimage[i], orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
        fv.append(a)
        hog_image.append(b)
    return fv,hog_image

#fonction pour l'etiquetage 
def etiq(listimage, etiquette) :
    etiq = []
    for i in range(len(listimage)) :
        etiq.append(etiquette)
    return etiq

avertissement_etiq = etiq(fv_avertissement,0)
obligation_etiq = etiq(fv_obligation,1)
interdiction_etiq = etiq(fv_interdiction,2)

### -----pretaitement de notre data set-------

#Chargement de nos images dans un vecteur 
images_obligation = Load_images("Traffic-routier/Obligation")
images_interdiction = Load_images("Traffic-routier/Interdiction_restriction")
images_avertissement = Load_images("Traffic-routier/Avertissement")
# Application d'un filtre Median 
for (i,j,k) in zip(range(len(images_avertissement)),range(len(images_interdiction)),range(len(images_obligation))) :
    images_avertissement[i] = cv.medianBlur(images_avertissement[i],3)
    images_interdiction[j] = cv.medianBlur(images_interdiction[j],3)
    images_obligation[k] = cv.medianBlur(images_obligation[k],3)
#Conversion de notre vecteur d'images RGB vers Y’CbCr
Ycbcr_obligation = Conversion_to_YCbCr(images_obligation)
Ycbcr_interdiction = Conversion_to_YCbCr(images_interdiction)
Ycbcr_avertissement = Conversion_to_YCbCr(images_avertissement)
#Redimensionement des images
resize_images(Ycbcr_avertissement)
resize_images(Ycbcr_obligation)
resize_images(Ycbcr_interdiction)
# Calcul du HOG pour notre vecteur d'images
fv_avertissement,hog_image_avertissement = Calcul_hog(Ycbcr_avertissement)
fv_obligation,hog_image_obligation = Calcul_hog(Ycbcr_obligation)
fv_interdiction,hog_image_interdiction = Calcul_hog(Ycbcr_interdiction)
# une fonction pour l'étiquetage
# dataset
fv = fv_avertissement + fv_obligation + fv_interdiction
etiq = avertissement_etiq + obligation_etiq + interdiction_etiq
# conversion en dataframe
fv = pd.DataFrame(fv)
etiq = np.array(etiq)

 ##--------Classification des images---------------##

##on utilise l'algorithme KPPV 
fv_train, fv_test, etiq_train, etiq_test = train_test_split(fv, etiq, test_size=0.8,random_state=42) 
print("Le nombre d'échantillon d'entrainement = ",fv_train.shape)
print("Le nombre d'échantillon de test = ",fv_test.shape)
K=5
# la mesure de distance est mise par défaut et c'est donc une distance Euclidienne
voisin = KNeighborsClassifier(n_neighbors = K, p = 2).fit(fv_train,etiq_train)
ypredict = voisin.predict(fv_test)
# Maintenant on passe à la phase prédiction sur l'ensemble de test
ypredict = voisin.predict(fv_test)
#Maintenant on passe au calcul de performance de notre modèle avec le calcul de certaines métriques en utilisant aussi la librairie sklearn
print("Taux de précision lors de l'entrainement : ",metrics.accuracy_score(etiq_train, voisin.predict(fv_train)))
print("Taux de précision lors des tests : ",metrics.accuracy_score(etiq_test, ypredict))
#Observons maintenant l'impact de la variation des paramètres en faisant la combinaison de toutes les possibilités avec les trois mesures de distances ainsi que la variation de k (k=1,3,5,7 et 9)
train_test = []
for k in range(1,10,2) :
    for i in range(1,4) : 
        voisin = KNeighborsClassifier(n_neighbors = k, p = i).fit(fv_train,etiq_train)
        ypredict = voisin.predict(fv_test)
        train_test.append(metrics.accuracy_score(etiq_test, ypredict))
print(train_test)
plt.figure(figsize=(6,6))
plt.plot(range(0,15),train_test,'g', label='Taux de précision')
plt.title("Graphique indiquant l'impact du changement des paramètres")
plt.legend(loc="upper right")
plt.ylabel("Taux de précision")
plt.xlabel("(nombres de voisin, distance choisie)")
plt.tight_layout()
plt.show()
# Calcul de la matrice de confusion dans le cas d'un voisin et d'une distance Euclidienne
voisin = KNeighborsClassifier(n_neighbors = 1, p = 2).fit(fv_train,etiq_train)
ypredict = voisin.predict(fv_test)
print("Matrice de confusion : \n",metrics.confusion_matrix(etiq_test, ypredict))
#reseaux de neurones 
accuracy_train = []
accuracy_test = []
for i in range(1,8) :
    modele = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(i), random_state=1).fit(fv_train,etiq_train)
    accuracy_train.append(metrics.accuracy_score(etiq_train, modele.predict(fv_train)))
    accuracy_test.append(modele.score(fv_test,etiq_test))
print(accuracy_train)
print(accuracy_test)
plt.figure(figsize=(6,6))
plt.plot(range(0,7),accuracy_test,'g', label='Taux de précision')
plt.title("Graphique indiquant l'impact du changement du nombre de neurones")
plt.legend(loc="lower right")
plt.ylabel("Taux de précision")
plt.xlabel("Nombre de neurones dans la couche cachée")
plt.tight_layout()
plt.show()
#Et par conséquent on choisit notre modèle avec 4 neurones dans la couche cachée, qui nous donne un résultat de 100% dans la phase de test
modele = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4), random_state=1).fit(fv_train,etiq_train)
print("Taux de précision lors de l'entrainement : ",metrics.accuracy_score(etiq_train, modele.predict(fv_train)))
modele.score(fv_test,etiq_test)
print("Les coefficients du réseaux de neurones : \n",modele.coefs_)
print("Les coefficients de la couche cachée : \n",modele.coefs_[1])
#On voit maintenant la matrice de confusion de notre modèle 
ymodele = modele.predict(fv_test)
print("Matrice de confusion : \n",metrics.confusion_matrix(etiq_test, ymodele))


# On va comparer la liste des résultats retrouvés avec le modèle et les prédictions 
print("classes théoriques : \n",etiq_test)
print("classes trouvées avec le modèle : \n",ypredict)
malclasse = []
for i in range(len(ypredict)) :
    if ypredict[i] != etiq_test[i] :
        malclasse.append(i)
print("les indices des erreurs sont : \n",malclasse)

# Maintenant on essaye de retrouver les images à qui cela appartient :
vecteurmalclasse = []
fv_test_tab = fv_test.to_numpy()
fv_tab = fv.to_numpy()
for i in range(len(malclasse)) :
    vecteurmalclasse.append(fv_test_tab[malclasse[i]])
print("Les vecteurs correspondants sont : \n",vecteurmalclasse)

# Comparons les valeurs de ces vecteurs avec les valeurs dans le dataset pour retrouver les indices des images
indices_imagesmal_classee = []
for i in range(len(fv_tab)) :
    for j in range(len(vecteurmalclasse)) :
        vrai = 1
        for l in range(len(vecteurmalclasse[j])) :
            if fv_tab[i][l] != vecteurmalclasse[j][l] :
                vrai = 0
        if vrai == 1 :  
            indices_imagesmal_classee.append(i) 
print("les indices des images mal classées dans le dataset de départ sont : ",indices_imagesmal_classee)
print("sachant que le dataset est classé de tels façon à avoir la disposition suivante : \n 50 premiers avertissement ensuite 50 obligation et ensuite 50 interdiction")
print("Donc les indices sont les indices de plaques d'obligations suivants : ")
for i in range(len(indices_imagesmal_classee)) : 
    print(indices_imagesmal_classee[i]-50)
print("Qui représente les images suivantes : ")
# On a donc les indices pour les images 
plt.figure(figsize=(21,7))
plt.subplot(1,4,1)
plt.axis("off")
plt.imshow(images_obligation[indices_imagesmal_classee[0]-50])
plt.title("Image 1")

plt.subplot(1,4,2)
plt.axis("off")
plt.imshow(images_obligation[indices_imagesmal_classee[1]-50])
plt.title("Image 2")

plt.subplot(1,4,3)
plt.axis("off")
plt.imshow(images_obligation[indices_imagesmal_classee[2]-50])
plt.title("Image 3")

plt.subplot(1,4,4)
plt.axis("off")
plt.imshow(images_obligation[indices_imagesmal_classee[3]-50])
plt.title("Image 4")
plt.show()