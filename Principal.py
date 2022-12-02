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


st.title('Saúde Materna')
Age = st.number_input('Digite a idade')
SystolicBP = st.number_input('Digite a pressão sistólica')
DiastolicBP = st.number_input('Digite a pressao diastólica')
BodyTemp = st.number_input('Digite a temperatura corporal')
HeartRate = st.number_input('Digite a frequência cardíaca')
if st.button('Clique aqui'):
  resultado = model.predict([[Age,SystolicBP,DiastolicBP,BodyTemp,HeartRate]])
  
  if resultado == ('high risk'):
    st.write('Alto Risco')

  if resultado == ('mid risk'):
    st.write('Médio Risco')

  if resultado == ('low risk'):
    st.write('Baixo Risco')
  
  
