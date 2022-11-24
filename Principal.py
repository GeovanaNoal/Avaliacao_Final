import streamlit as st

from sklearn.naive_bayes import GaussianNB
import pandas as pd
dados = pd.read_csv('Dados de Risco à Saúde Materna.csv')




classes = dados['RiskLevel']
nomesColunas = dados.columns.to_list()
tamanho = len(nomesColunas)
nomesColunas = nomesColunas[1:tamanho-1]
features = dados[nomesColunas]

from sklearn.model_selection import train_test_split

features_treino,features_teste,classes_treino,classes_teste = train_test_split(features,
                                                                               classes,
                                                                               test_size=0.26,
                                                                               random_state=3)

model = GaussianNB() 

model.fit(features_treino,classes_treino)
predicoes = model.predict(features_teste)


st.title('Aplicativo de IA')
Age = st.number_input('Digite a idade')
SystolicBP = st.number_input('Digite a pressao sistolica')
DiastolicBP = st.number_input('Digite a pressao diastolica')
BodyTemp = st.number_input('Digite a temperatura corporal')
HeartRate = st.number_input('Digite a frequencia cardiaca')
if st.button('Clique aqui'):
  resultado = model.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
  
  if resultado == ('Iris-setosa'):
    st.write('setosa')
    st.image('iris_setosa.jpg')
    
  if resultado == ('Iris-versicolor'):
    st.write('versicolor')
    st.image('iris_versicolor.jpg')
   
  if resultado == ('Iris-virginica'):
    st.write('virginica')
    st.image('iris_virginica.jpg')
