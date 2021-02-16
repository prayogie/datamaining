
import inline as inline
import matplotlib
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import Sastrawi.Stemmer
import string
from nltk.tokenize import word_tokenize
from string import digits
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

nltk.download('punkt')

sundanese = pd.read_csv('datasetmaining.csv')

fitur = sundanese.iloc[:,0].values
labels = sundanese.iloc[:,1].values

factory = StemmerFactory()
stemmer = factory.create_stemmer()

fitur_ekstraksi0 = []
for cuitan in range(0, len(fitur)):
    tmp = str(fitur[cuitan]).lower()
    fitur_ekstraksi0.append(tmp)


fitur_ekstraksi1 = []
for cuitan in range(0, len(fitur)):
    tmp = fitur_ekstraksi0[cuitan].translate(str.maketrans(' ',' ', digits)) # membuang karakter angka
    fitur_ekstraksi1.append(tmp)

fitur_ekstraksi2 = []
for cuitan in range(0, len(fitur_ekstraksi1)):
    tmp = fitur_ekstraksi1[cuitan].translate(str.maketrans(' ',' ', string.punctuation)) # membuang karakter
    fitur_ekstraksi2.append(tmp)


fitur_ekstraksi3 = []
# fitur_ekstraksi = []
for cuitan in range(0, len(fitur_ekstraksi2)):
    tmp = re.sub(r'\W', ' ',str(fitur_ekstraksi2[cuitan])) # membuang karakter khusus selain angka dan huruf
    tmp = re.sub(r'\s+[a-zA-Z]\s+', ' ',str(fitur_ekstraksi2[cuitan])) # membuang kata yang hanya satu huruf
    tmp = re.sub(r'\^[a-zA-Z]\s+', ' ',str(fitur_ekstraksi2[cuitan])) # membuang kata yang hanya satu huruf dari awal
    tmp = re.sub(r'\s+', ' ',str(fitur_ekstraksi2[cuitan])) # mengganti spasi ganda dengan spasi tunggal
    fitur_ekstraksi3.append(tmp)
    # fitur_ekstraksi.append(tmp)


# fitur_ekstraksi4 = []
# for cuitan in range(0, len(fitur_ekstraksi3)):
#     tmp = stemmer.stem(str(fitur_ekstraksi3[cuitan]))
#     fitur_ekstraksi4.append(tmp)

fitur_ekstraksi5 = []
for cuitan in range(0, len(fitur_ekstraksi3)):
    tmp = word_tokenize(str(fitur_ekstraksi3[cuitan]))
    fitur_ekstraksi5.append(tmp)


stopsunda1 = open('stopwateindo.txt', 'r')
stopsunda2 = stopsunda1.read()
stopsunda = word_tokenize(stopsunda2)

fitur_ekstraksi = []
def swr(a, b):
    filtered_sentence = []
    for w in a:
        if w not in b:
            filtered_sentence.append(w)
    return filtered_sentence

for cuitan in range(0, len(fitur_ekstraksi5)):
    tmp = swr(fitur_ekstraksi5[cuitan], stopsunda)
    fitur_ekstraksi.append(tmp)
# print(*fitur_ekstraksi[:5], sep='\n')
def identity_tokenizer(text):
    return text

from sklearn.feature_extraction.text import TfidfVectorizer

vektor_kata = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
fitur_ekstraksi = vektor_kata.fit_transform(fitur_ekstraksi).toarray()
# print(fitur_ekstraksi[:1])
#
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(fitur_ekstraksi, labels, train_size=0.8, random_state=0)

from sklearn.ensemble import RandomForestClassifier

klasifier = RandomForestClassifier(n_estimators=200, random_state=0)
klasifier.fit(X_train, y_train)

hasil_prediksi = klasifier.predict(X_test)
# print(*hasil_prediksi[:5])
# print('\n')

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test,hasil_prediksi))
print(classification_report(y_test,hasil_prediksi))
print(accuracy_score(y_test, hasil_prediksi))

from sklearn.metrics import plot_confusion_matrix
titles_options = [("Confusion matrix", None)]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(klasifier, X_test, y_test,
                                 display_labels=['sms normal', 'penipuan', 'promo'],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize,
                                 values_format='g')
    disp.ax_.set_title(title)

plt.show()