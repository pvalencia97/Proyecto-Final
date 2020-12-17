#importamos todas las funciones y librerias a utilizar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.discriminant_analysis as da
import sklearn.neighbors as nb
from sklearn.metrics import confusion_matrix, classification_report, precision_score, accuracy_score
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

#datos del dataframe
Weekly = pd.read_csv('Weekly.csv')
Weekly.head()

#mostrar la correlacion que existe entre los predictores
Weekly.corr()

#Grafica para los predictores 'Year' y 'Volume'
plt.clf()
plt.title('Grafica de los predictores Year y Volume')
plt.figure(1,figzise=(6,8))
plt.xlabel('Year')
plt.ylabel('Volume')
x = Weekly.Year
y = Weekly.Volume
plt.scatter(x,y,color='r')
plt.show()

#descripcion del conjunto de datos
Weekly.describe()
Weekly.median()#mediana
Weekly.cov()#matriz de covarianza
Weekly.mode()#moda

#resultados de direction 
Weekly.groupby('Direction').size()

#histogramas de los predictores
Weekly.drop(['Direction'],1).hist()
plt.show()

#sns.pairplot(Weekly.dropna(), hue='Direction',size=8,vars=["Year", "Lag1","Lag2","Lag3","Lag4","Lag5","Volume","Today"],kind='reg')


#-------------------REGRESION LOGISTICA--------------------##
#realizamos modelo de regresion logistica
modelo = smf.glm( formula = 'Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume', data = Weekly, family = sm.families.Binomial() ).fit()
modelo.summary()#mostramos resultados del modelo
Weekly['Direction2'] = Weekly.Direction.factorize()[0]
y = Weekly.Direction2
X = Weekly[['Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5','Volume']]
X = sm.add_constant(X)
y_prob = modelo.predict(X)
decision_prob = 0.5
comparar_results = pd.DataFrame({'True default status': y,'Predicted default status': y_prob > decision_prob})
comparar_results.replace(to_replace={True:1, False:0}, inplace=True)
y_pred = np.asarray(comparar_results['Predicted default status'])

comparar_results.replace(to_replace={0:'Up', 1:'Down'}, inplace=True)

#comparar_results.groupby(['Predicted default status','True default status']).size().unstack('True default status')
confusion_matrix(y, y_pred)
precision_score(y, y_pred)
accuracy_score(y, y_pred)

#(d)

#conjunto de entrenamiento
X_train = Weekly[Weekly.Year < 2009][['Lag2']]
y_train = Weekly[Weekly.Year < 2009][['Direction2']]
#conjunto de prueba
X_test = Weekly[Weekly.Year >= 2009][['Lag2']]
y_test = Weekly[Weekly.Year >= 2009][['Direction2']]
X_train = sm.add_constant(X_train) 
X_test = sm.add_constant(X_test)

modelo2 = smf.Logit(y_train, X_train).fit()
modelo2.summary()

#para el conjunto de entrenamiento
y_prob_train = modelo2.predict(X_train)
decision_prob = 0.5

y_train = y_train.reset_index()['Direction2']
y_prob_train = y_prob_train.reset_index()[0]
comparar_results = pd.DataFrame({'True default status': y_train,
                    'Predicted default status': y_prob_train > decision_prob})
comparar_results.replace(to_replace={True:1, False:0}, inplace=True)
y_pred_train = np.asarray(comparar_results['Predicted default status'])
comparar_results.replace(to_replace={0:'Up', 1:'Down'}, inplace=True)
precision_score(y_train, y_pred_train)
accuracy_score(y_train, y_pred_train)
confusion_matrix(y_train, y_pred_train).T
#precision_score(y_test, y_pred_test)
#accuracy_score(y_test, y_pred_test)
#confusion_matrix(y_test,y_pred_test).T

#para el conjunto de prueba
y_prob_test = modelo2.predict(X_test)
decision_prob = 0.5
y_test = y_test.reset_index()['Direction2']
y_prob_test = y_prob_test.reset_index()[0]
comparar_results = pd.DataFrame({'True default status': y_test,
                    'Predicted default status': y_prob_test > decision_prob})
comparar_results.replace(to_replace={True:1, False:0}, inplace=True)
y_pred_test = np.asarray(comparar_results['Predicted default status'])
comparar_results.replace(to_replace={0:'Up', 1:'Down'}, inplace=True)
#comparar_results.groupby(['Predicted default status','True default status']).size().unstack('True default status')
precision_score(y_test, y_pred_test)
accuracy_score(y_test, y_pred_test)
confusion_matrix(y_test,y_pred_test).T

#-------------------MODELO LDA--------------------##

#conjunto de entrenamiento
X_train = Weekly[Weekly.Year < 2009][['Lag2']]
y_train = Weekly[Weekly.Year < 2009].Direction2.factorize()[0]
#conjunto de prueba
X_test = Weekly[Weekly.Year >= 2009][['Lag2']]
y_test = Weekly[Weekly.Year >= 2009].Direction2.factorize()[0]
#X_train = sm.add_constant(X_train) 
#X_test = sm.add_constant(X_test)

#definimos modelo LDA
lda = da.LinearDiscriminantAnalysis()

#para el conjunto de entrenamiento
lda.fit(X_train, y_train)
y_pred_train = lda.predict(X_train)
lda.priors_
lda.means_
lda.coef_

precision_score(y_train, y_pred_train)
accuracy_score(y_train, y_pred_train)
confusion_matrix(y_train, y_pred_train).T
#confusion_matrix(y_train, y_pred).T

#para el conjunto de prueba
#lda.fit(X_test,y_test)
y_pred_test = lda.predict(X_test)
lda.priors_
lda.means_
lda.coef_
precision_score(y_test, y_pred_test)
accuracy_score(y_test, y_pred_test)
confusion_matrix(y_test,y_pred_test).T

#-------------------MODELO QDA--------------------##
qda = da.QuadraticDiscriminantAnalysis()

#para el conjunto de entrenamiento
y_pred_train = qda.fit(X_train, y_train).predict(X_train)
qda.priors_
qda.means_
precision_score(y_train, y_pred_train)
accuracy_score(y_train, y_pred_train)
confusion_matrix(y_train, y_pred_train).T
#para el conjunto de prueba
#para el conjunto de prueba
y_pred_test = qda.predict(X_test)
qda.priors_
qda.means_
precision_score(y_test, y_pred_test)
accuracy_score(y_test, y_pred_test)
confusion_matrix(y_test,y_pred_test).T

##knn
#para el conjunto de entrenamiento
knn1 = nb.KNeighborsClassifier(n_neighbors=1)
knn1_model = knn1.fit(X_train, y_train)
pred1_train = knn1_model.predict(X_train)

precision_score(y_train, pred1_train)
accuracy_score(y_train, pred1_train)
confusion_matrix(y_train, pred1_train).T

#para el conjunto de pruebas
#knn_1_test = nb.KNeighborsClassifier(n_neighbors=1)
#knn_1_model_test = knn_1_train.fit(X_test, y_test)
pred1_test = knn1_model.predict(X_test)
precision_score(y_test, pred1_test)
accuracy_score(y_test, pred1_test)
confusion_matrix(y_test, pred1_test).T
