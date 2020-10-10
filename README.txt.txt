PART A

1- B
2- B
3- B
4- D
5- A
6- D
7- C
8- D
9- D
10- C
11- B
12- D
13- C
14- B
15- C
16- B
17- A
18- C
19- D
20- B
21- A
22- B
23- C
24- C
25- A



PART B

1- decrease
2- non-linear classifier, decrease
3- classification
4- testing, training
5- we will normalize data 0,1





PART C



Q.1

string = "Steve was born in Tokyo, Japan in 1950. He moved to London with his parents when he was 5 years old. Steve started school there and his father began work at the hospital. His mother was a house wife and he had four brothers. He lived in England for 2 years then moved to Amman, Jordan where he lived there for 10 years. Steve then moved to Cyprus to study at the Mediterranean University. Unfortunately, he did not succeed and returned to Jordan. His parents were very unhappy so he decided to try in America. He applied to many colleges and universities in the States and finally got some acceptance offers from them. He chose Wichita State University in Kansas. His major was Bio-medical Engineering. He stayed there for bout six months and then he moved again to a very small town called Greensboro to study in a small college."
print(string)

final_string = string.lower()
print(final_string)

def removePunctuation(string): 
    puncts = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
 
    for x in string.lower(): 
        if x in puncts: 
            string = string.replace(x, "") 
  
    # Print string without punctuation 
    print(string) 
string = "Steve was born in Tokyo, Japan in 1950. He moved to London with his parents when he was 5 years old. Steve started school there and his father began work at the hospital. His mother was a house wife and he had four brothers. He lived in England for 2 years then moved to Amman, Jordan where he lived there for 10 years. Steve then moved to Cyprus to study at the Mediterranean University. Unfortunately, he did not succeed and returned to Jordan. His parents were very unhappy so he decided to try in America. He applied to many colleges and universities in the States and finally got some acceptance offers from them. He chose Wichita State University in Kansas. His major was Bio-medical Engineering. He stayed there for bout six months and then he moved again to a very small town called Greensboro to study in a small college."

removePunctuation(string)

def removeWhitespace(string): 
    return string.replace(" ", "") 
    
string = "Steve was born in Tokyo, Japan in 1950. He moved to London with his parents when he was 5 years old. Steve started school there and his father began work at the hospital. His mother was a house wife and he had four brothers. He lived in England for 2 years then moved to Amman, Jordan where he lived there for 10 years. Steve then moved to Cyprus to study at the Mediterranean University. Unfortunately, he did not succeed and returned to Jordan. His parents were very unhappy so he decided to try in America. He applied to many colleges and universities in the States and finally got some acceptance offers from them. He chose Wichita State University in Kansas. His major was Bio-medical Engineering. He stayed there for bout six months and then he moved again to a very small town called Greensboro to study in a small college."
print(removeWhitespace(string)) 


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
  
string = "Steve was born in Tokyo, Japan in 1950. He moved to London with his parents when he was 5 years old. Steve started school there and his father began work at the hospital. His mother was a house wife and he had four brothers. He lived in England for 2 years then moved to Amman, Jordan where he lived there for 10 years. Steve then moved to Cyprus to study at the Mediterranean University. Unfortunately, he did not succeed and returned to Jordan. His parents were very unhappy so he decided to try in America. He applied to many colleges and universities in the States and finally got some acceptance offers from them. He chose Wichita State University in Kansas. His major was Bio-medical Engineering. He stayed there for bout six months and then he moved again to a very small town called Greensboro to study in a small college."
  
stopWords = set(stopwords.words('english')) 
  
wordTokens = word_tokenize(string) 
  
filteredSentence = [w for w in wordTokens if not w in stopWords] 
  
filteredSentence = [] 
  
for w in wordTokens: 
    if w not in stopWords: 
        filteredSentence.append(w) 
  
print(wordTokens) 
print(filteredSentence) 


import nltk
from nltk.stem.porter import PorterStemmer
porter_stemmer = PorterStemmer()

word_data = "Steve was born in Tokyo, Japan in 1950. He moved to London with his parents when he was 5 years old. Steve started school there and his father began work at the hospital. His mother was a house wife and he had four brothers. He lived in England for 2 years then moved to Amman, Jordan where he lived there for 10 years. Steve then moved to Cyprus to study at the Mediterranean University. Unfortunately, he did not succeed and returned to Jordan. His parents were very unhappy so he decided to try in America. He applied to many colleges and universities in the States and finally got some acceptance offers from them. He chose Wichita State University in Kansas. His major was Bio-medical Engineering. He stayed there for bout six months and then he moved again to a very small town called Greensboro to study in a small college."

nltk_tokens = nltk.word_tokenize(word_data)

for w in nltk_tokens:
    print ("Actual: %s  Stem: %s"  % (w,porter_stemmer.stem(w)))


import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

word_data = "Steve was born in Tokyo, Japan in 1950. He moved to London with his parents when he was 5 years old. Steve started school there and his father began work at the hospital. His mother was a house wife and he had four brothers. He lived in England for 2 years then moved to Amman, Jordan where he lived there for 10 years. Steve then moved to Cyprus to study at the Mediterranean University. Unfortunately, he did not succeed and returned to Jordan. His parents were very unhappy so he decided to try in America. He applied to many colleges and universities in the States and finally got some acceptance offers from them. He chose Wichita State University in Kansas. His major was Bio-medical Engineering. He stayed there for bout six months and then he moved again to a very small town called Greensboro to study in a small college."

nltk_tokens = nltk.word_tokenize(word_data)
for w in nltk_tokens:
       print "Actual: %s  Lemma: %s"  % (w,wordnet_lemmatizer.lemmatize(w))





Q.3

class Main 
{ 
    public static void modifyMatrix(int mat[ ][ ], int R, int C) 
    { 
        int row[ ]= new int [R]; 
        int col[ ]= new int [C]; 
        int i, j; 
      
        /* Initialize all values of row[] as 0 */
        for (i = 0; i < R; i++) 
        { 
        row[i] = 1; 
        } 
      
      
        /* Initialize all values of col[] as 0 */
        for (i = 0; i < C; i++) 
        { 
        col[i] = 1; 
        } 
      
      
        /* Store the rows and columns to be marked as 
        1 in row[] and col[] arrays respectively */
        for (i = 0; i < R; i++) 
        { 
            for (j = 0; j < C; j++) 
            { 
                if (mat[i][j] == 0) 
                { 
                    row[i] = 0; 
                    col[j] = 0; 
                } 
            } 
        } 
      
        /* Modify the input matrix mat[] using the 
        above constructed row[] and col[] arrays */
        for (i = 0; i < R; i++) 
        { 
            for (j = 0; j < C; j++) 
            { 
                if ( row[i] == 0 || col[j] == 0 ) 
                { 
                    mat[i][j] = 0; 
                } 
            } 
        } 
    } 
      
    /* A utility function to print a 2D matrix */
    public static void printMatrix(int mat[ ][ ], int R, int C) 
    { 
        int i, j; 
        for (i = 0; i < R; i++) 
        { 
            for (j = 0; j < C; j++) 
            { 
                System.out.print(mat[i][j]+ " "); 
            } 
            System.out.println(); 
        } 
    } 
      
    /* Driver program to test above functions */
    public static void main(String[] args)  
    { 
        int mat[ ][ ] = { {1, 1, 1, 1}, 
                          {0, 0, 1, 1}, 
                          {1, 1, 1, 1},}; 
                      
                System.out.println("Matrix Intially"); 
                  
                printMatrix(mat, 3, 4); 
              
                modifyMatrix(mat, 3, 4); 
                System.out.println("Matrix after modification n"); 
                printMatrix(mat, 3, 4); 
              
    }  
  
} 


Q.5

Collab Link for this question: https://colab.research.google.com/drive/1Wtcvy2M2_-_ZEW1RWZSrBN4PcV1hP5Rg?usp=sharing

#the code below runs on local machine

import random
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras import Sequential
from keras.layers import Dense

from google.colab import drive

drive.mount(‘/content/drive’)
path = "copied path"


data=pd.read_csv(path)

data.drop(['filename'],axis=1, inplace=True)
data.head()

data.count()

#Seperating target variable
X= data.iloc[:,0:6]
y= data.iloc[:,6]
ynew=pd.DataFrame()

for i in range(len(y)):
    if(y[i]=="noise"):
        y[i]=1
    else:
        y[i]=0

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
from keras import Sequential
from keras.layers import Dense
classifier = Sequential()
#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=6))
#Second  Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal'))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics =['accuracy'])

#Fitting the data to the training dataset
classifier.fit(X_train,y_train, batch_size=10, epochs=100)

y_pred=classifier.predict(X_test)
y_pred =(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

