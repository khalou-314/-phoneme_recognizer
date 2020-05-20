from scipy import signal
import math
from math import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import wave
import cmath as cm
import sys
import os


# Permet de générer un .csv pour enregistrer les courbes déjà calculées
#Finalement on n'utilise pas de .csv mais on calcule notre db a chaque fois
def baseDeDonnee(file_name):
    db_H=[]
    for k in range(len(classes)):
        db_H.append([])
        for i in range(len(baseDonnees[k])):
            sig,n,H = getH(baseDonnees[k][i])
            db_H[k].append(H.copy())
    return db_H

# comparaison : permet de trouver la classe la plus proche
def comparaison(H,db_H,n):
    distMiniNormee = 1000000000 #Initialisation à une distance très grande
    distMiniBrute = 1000000000 #Idem
    listDistance = []
    indiceClassemin_normee = 0 
    indiceClassemin_brute = 0
    classe_min_moy = 0

    # Calcul des distances des points H avec l'ensemble des moyennes Hm
    for k in range(1,len(classes)): # Loop sur les différentes classes
        distanceMoyenne = 0
        for i in range(len(baseDonnees[k])): # Loop pour tous les fichiers de la classe k
            Hm = db_H[k][i]
            dmBrute = distBrut(H,Hm,n)
            dmNormee = distNormee(H,Hm,n)

            distanceMoyenne += dmNormee

            if dmBrute < distMiniBrute: # On garde la plus petite distance brute
                indiceClassemin_brute = k
                distMiniBrute = dmBrute

            if dmNormee < distMiniNormee:
                indiceClassemin_normee = k
                distMiniNormee = dmNormee

        distanceMoyenne = distanceMoyenne/len(baseDonnees[k])
        listDistance.append(distanceMoyenne)
    return indiceClassemin_normee, indiceClassemin_brute, listDistance.index(min(listDistance))


#Permet de calculer la distance brute entre H et Hm
def distBrut(H,Hm,n):
    dist = 0
    Kmax = int((S*n*fmax/fe)+0.5)
    Kmin = int(S*n*fmin/fe)
    for k in range (Kmin,Kmax):
        dist += abs(H[k]-Hm[k])
    return dist


# Calcule la distance normée entre H et Hm
def distNormee(H,Hm,n):
    dist = 0
    Kmax = int((S*n*fmax/fe)+0.5)
    Kmin = int(S*n*fmin/fe)
    moyenne_H = moyenneQuad(H,n)
    moyenne_Hm = moyenneQuad(Hm,n)
    for k in range (Kmin,Kmax):
        dist += pow(H[k]/moyenne_H - Hm[k]/moyenne_Hm,2)
    return sqrt(dist)


# Donne la moyenne quadratique de H
def moyenneQuad(H,n):
    moy = 0
    Kmax = int((S*n*fmax/fe)+0.5)
    Kmin = int(S*n*fmin/fe)
    for k in range(Kmin,Kmax):
        moy += pow(H[k],2)  
    return sqrt( moy/(Kmax-Kmin))


# Signal echantilloné de x
def sigEchX(title):
    w = wave.open(title, 'rb')
    sig = w.readframes(-1)
    sig =  np.fromstring(sig, "Int16")
    N = 32768 # valeur mini du signal
    if len(sig) < N :
        print("erreur trop court !")
        return [],len(sig)
    sig= sig[:N]
    return sig, N



# Calcul le signal fenetré
def sigFenetre(x,n):
    w_ = [ (1 + cos(2*pi*(k-n/2)/n))/2 for k in range (n)] # hamming simplifié
    x_ = [ x[i] * w_[i] for i in range(n)] # sig*hamming
    return x_



# Spectre du signal fenétré et suréchantillonné S fois
def sigSurech(x,n):
    X = np.fft.fft(x,n*S)
    X = X[:n//2]
    return abs(X) # remplacer par dft si possible

#Extraction de la partie utile du spectre
def utileX(X,n):
    Kmax = int((S*n*fmax/fe)+0.5)
    Kmin = int(S*n*fmin/fe)
    X_ = X[:Kmax]
    for i in range(Kmin):
        X_[i]=0
    return X_



# Obtenir le spectre accentué
def spectreAccentue(X,n):
    kc = S*n*fc/fe
    P = [ sqrt(1 + pow(k/kc,2*alpha)) for k in range (len(X))] # module d'un filtre passe-haut Butterworth
    Y = [ P[k]*X[k] for k in range(len(X))]
    return Y

# butterWorth, lissage par filtrage

def lissageY(Q,Y,n):
    N = len(Y)
    Y_ = np.zeros(N)
    Kmax = int((S*n*fmax/fe)+0.5)
    Kmin = int(S*n*fmin/fe)
    for k in range (N):
        for q in range(-Q//2,Q//2):
            if ((k-q) >= Kmin) & ((k-q) < Kmax):
                Y_[k] += w_(q) * Y[k-q]
    return Y_

#Fenetre de Hamming
def w_(q):
    return (1+cos(2*pi*q/Q))/2

#Calcul du pitch
def pitch(X):
    n = len(X)
    dmin = inf
    m = 150
    min = 100
    max = 900
    for m in range(min,max):
        d = Damdf(m,X,n-m)
        if dmin > d :
            dmin = d
            m_min = m
    return m_min

#
def Damdf(m,X,N):
    dm = 0
    for  n in range(N):
        dm += abs(X[n] - X[n+m])
    dm = dm/N
    return dm


# Pour obtenir x(t), nombre échantillons, hauteur, H(f)
def getH (file,show_graph = False):
    #1.a Lecture fichier .wav : on obtient x(t) et on le fenêtre pour avoir x'(t)
    x,n = sigEchX(file)
    #1.b Obtention Spectre suréchantilloné : on obtient x'(t)
    x_ = sigFenetre(x,n)
    #2.a Calcule son spectre : on obtient X''(f)
    X = sigSurech(x,n)
    # 2.b On enleve les niveaux inf et sup: on obtient X'''(f)
    X_ = utileX(X,n)
    # 3.a Accentuation du spectre : Y'(f)
    Y = spectreAccentue(X_,n)
    # 3.b Lisser le spectre : Y''(f)
    #Y_ = lissageY(Q,Y,n)
    #Kmin = int(S*n*fmin/fe)
    #Q=(pitch(X_)//2)
    gaussian = signal.gaussian(len(Y),Q)
    Y_ = signal.fftconvolve(Y, gaussian,mode='same')
    # 5. H(f) environ égal à Y''(f)
    H = Y_
    temps = np.arange(n,)*Te
    frequences = np.arange(len(X),)*fmax/(len(X))
    frequences_ = np.arange(len(X_),)*fmax/(len(X_))

    #display(frequences_,gaussian,'gaussian',1)
    if show_graph :
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(temps, x, 'tab:purple')
        axs[0, 0].set_title("signal x(t) reçu")
        axs[0, 1].plot(temps, x_, 'tab:orange')
        axs[0, 1].set_title("signal x''(t) reçu")
        axs[1, 0].plot(frequences_,X_, 'tab:red')
        axs[1, 0].set_title("spectre X'''(f) du signal")
        axs[1, 1].plot(frequences_,Y_, 'tab:green')
        axs[1, 1].set_title("spectre Y''(f) du signal")

        for ax in axs.flat:
            ax.set(xlabel='Fréquence', ylabel='Amplitude')
        for ax in axs.flat:
            ax.label_outer()
        plt.show()

    return x,n,H

##
# graphe
def display(X,Y,titre,i):
    plt.plot(X,Y)
    plt.figure(i)
    plt.title(titre)
    plt.show()
    return

# Paramètres #
fe = 44100
Te = 1/fe
n = 32768
S = 4
fmin = 150
fmax = 2200
fc = 1200
alpha = 5
Q = 150 # A modifier

#Fichier à inclure dans notre base de données commune
classes = ['a','e','é','è','i','o','ou','an','in','on','u','m','n']
baseDonnees = [           
['c_a.wav',
#'k_a.wav',
'l_a.wav',
't_a.wav',
'm_a.wav'
],
[#'k_e.wav',
'c_e.wav',
'l_e.wav',
't_e.wav',
'm_e.wav'
],
[#'k_é.wav',
'c_é.wav',
'l_é.wav',
't_é.wav',
'm_é.wav'
],
[#'k_è.wav',
'c_è.wav',
'l_è.wav',
't_è.wav',
'm_è.wav'
],
[#'k_i.wav',
'c_i.wav',
'l_i.wav',
't_i.wav',
'm_i.wav'
],
[#'k_o.wav',
'c_o.wav',
'l_o.wav',
't_o.wav',
'm_o.wav'
],
[#'k_ou.wav',
'c_ou.wav',
'l_ou.wav',
't_ou.wav',
'm_ou.wav'
],
[#'k_an.wav',
'c_an.wav',
'l_an.wav',
't_an.wav',
'm_an.wav'
],
[#'k_in.wav',
'c_in.wav',
'l_in.wav',
't_in.wav',
'm_in.wav'
],
[#'k_on.wav',
'c_on.wav',
'l_on.wav',
't_on.wav',
'm_on.wav'
],
[#'k_u.wav',
'c_u.wav',
'l_u.wav',
't_u.wav',
'm_u.wav'
],
[#'k_m.wav',
'c_m.wav',
'l_m.wav',
't_m.wav',
'm_m.wav'
],
[#'k_n.wav',
'c_n.wav',
'l_n.wav',
't_n.wav',
'm_o.wav'
]
]

#Pour choper le chemin 
print(os.getcwd())

main_path = os.getcwd()
for i in range(len(baseDonnees)):
    for j in range(len(baseDonnees[i])):
        new_name = str(main_path +"\\resources\\"+baseDonnees[i][j])
        baseDonnees[i][j] = new_name


#Main
# Génération de la base de données
db_csv = "db.csv"
db_H = baseDeDonnee(db_csv)

# Fichier à tester

entree = str(input("Entrer un fichier wav : "))
sig,N,H = getH(main_path +"\\resources\\"+entree,True) # chemin absolu du fichier à tester
norm,brut,moy = comparaison(H,db_H,N)
print("Phonème reconnu comme "+ classes[brut]+" par distance brut, "+classes[norm]+" par distance normée, " + classes[moy]+" par distance moyenne sur une classe")
