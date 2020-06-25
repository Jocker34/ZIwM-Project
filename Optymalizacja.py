import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tqdm import tqdm
from copy import deepcopy

#Wczytywanie danych
dataOfHepatitis = np.loadtxt('dane.txt', delimiter=",")
#print(dataOfHepatitis)

#Normalizacja danych
for line in dataOfHepatitis:
    #print(line)
    for value in range(len(line)):
        #print(line)
        #print(line[value])
        if(value == 1 or value == 16 or value == 18):
            line[value] /= 100
        if(value == 14 or value == 17):
            line[value] /= 10
        if(value == 15):
            line[value] /= 1000
        else:
            if(line[value] == 1):
                line[value] = 0
            if(line[value] == 2):
                line[value] = 1
    #print(line)

#Zapis znormalizowanych danych do nowego pliku 
np.savetxt("newData.txt", dataOfHepatitis, fmt="%s")
#print(dataOfHepatitis)

columWithClass = dataOfHepatitis[:,0]
allColumnsWithoutClass = dataOfHepatitis[:, 1:]

#Ranking cech
analysisOfVariance = SelectKBest(f_classif)
kBestData = analysisOfVariance.fit(allColumnsWithoutClass, columWithClass)
featuresRanking = analysisOfVariance.scores_

print(featuresRanking)

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix

# Przechowywanie best score, momentum, layerSize, featureCount, confusion matrix
BestScore = [0, True, 0, 0, np.ndarray]
ListScore = []


def classification(X=allColumnsWithoutClass, y=columWithClass, relu=True, layerSize=100, bestScore=BestScore, featureCount=4):
    fvalue_selector = SelectKBest(f_classif, k=featureCount)  # f. licząca wartość analizy wariacji + ilości cech
    X_reduced = fvalue_selector.fit_transform(X, y)  # redukcja do wybranej ilości cech

    # Tworzenie wielokrotnego 'K-Fold cross validator' podział na 2 grupy do trenowania
    # i testów. Powtórzony 5 razy, z ziarnem podziału 2652124
    rkf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=2652124)

    if relu:
        mlp = MLPClassifier(hidden_layer_sizes=layerSize, solver='adam', max_iter=1000)
    else:
        mlp = MLPClassifier(hidden_layer_sizes=layerSize, activation='logistic', solver='adam', max_iter=1000)

    # Pętla dla 5 krotnej walidacji krzyżowej (train porusza sie po indeksach grupy 1(trenujacej),test po 2(testowej))
    for train, test in rkf.split(X_reduced, y):
        x_train, x_test = X_reduced[train], X_reduced[test]  # Przypisanie list x_train, x_test
        y_train, y_test = y[train], y[test]  # Przypisanie list y_train, y_test

        mlp.fit(x_train, y_train)  # Dopasowanie x_train do y_train

        # Przypisanie 'mean accuracy on the given test data' na podstawie nowych zbiorów
        score = mlp.score(x_test, y_test)

        predict = mlp.predict(x_test)  # Przewidywanie stworzonym mlp przy uzyciu zbioru x_test

        # Macierz pomyłek y_test zawiera prawidłowe klasy, predict zawiera przewidziane
        confusionMatrix = confusion_matrix(y_test, predict)

        # print(confusionMatrix)
        ListScore.append([score, relu, layerSize, featureCount, confusionMatrix])
        if bestScore[0] < score:  # Przypisanie największej liczby punktów do listy
            bestScore = [score, relu, layerSize, featureCount, confusionMatrix]
            BestScore = deepcopy(bestScore)

    return bestScore  # Zwrócenie największej ilości punktów


layerS = [100, 200, 500]
function = [True, False]

for layer in layerS:
    for func in function:
        for i in tqdm(range(1, 9
                            )):
            BestScore = classification(relu=func, layerSize=layer, bestScore=BestScore, featureCount=i)

print(BestScore)
dflist = pd.DataFrame(ListScore)
dflist.to_csv('Wyniki.csv', encoding='utf-8', index=False)
