import numpy as np
import pandas as pd
import re
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix





dataset = pd.read_csv("logs.csv")

def limpar_texto(texto, n = 2):
    texto_string = str(texto).lower()
    texto_limpo = re.sub(r'[^a-z0-9áàâãéêíóôõúç\s]+', '', texto_string)
    
    tokens = texto_limpo.split(' ')
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]




dataset["log_correto"] = dataset['log'].apply(limpar_texto)
 

X = dataset['log_correto']
Y = dataset['label']

sentencasTreino, sentecasTeste, respostasTreino, respostasTeste = train_test_split(X, Y, test_size = 0.2, random_state = 42)

w2v_model = Word2Vec(
    sentences=sentencasTreino,
    vector_size=100,
    window=5,
    min_count=1,
    workers=1,
    seed=42 ,
    epochs=75
)

def vetorMedio(X, w2v_model, vetores_size = 100):
    for tok in X:
        if tok  in w2v_model.wv:
            vetores = [w2v_model.wv[tok]]


    return np.mean(vetores, axis = 0)

        

# pegar os vetores e empilhar usando vstack
X_w2v_train = np.vstack([vetorMedio(tokens, w2v_model, 100) for tokens in sentencasTreino])
X_w2v_test = np.vstack([vetorMedio(tokens, w2v_model, 100) for tokens in sentecasTeste])



modelo = LocalOutlierFactor(novelty=True)

normais = (respostasTreino==0)
somenteNormais = X_w2v_train[normais]
modelo.fit(somenteNormais)
previsoes = modelo.predict(X_w2v_test)
previsoes_ajustadas = np.where(respostasTeste == 1, -1, 1)


print("Matriz de Confusão:")
print(confusion_matrix(previsoes_ajustadas, previsoes))

print("\nRelatório de Classificação:")
print(classification_report(previsoes_ajustadas, previsoes))


print("divisao dos modelos ")
print("#" * 50)

modelo2 = LocalOutlierFactor(contamination=0.05)

previsoes_sem_rotulo = modelo2.fit_predict(X_w2v_train)


df_analise = pd.DataFrame({
    'log_texto': sentencasTreino, 
    'classificacao_lof': previsoes_sem_rotulo
})

# filtra e exibe na tela apenas o que o modelo considerou anomalia (-1)
anomalias_detectadas = df_analise[df_analise['classificacao_lof'] == -1]

print(f"Total de anomalias encontradas: {len(anomalias_detectadas)}")
print("\nExemplo das anomalias detectadas:")
print(anomalias_detectadas.head(15))

print("divisao dos modelos ")
print("#" * 50)

modelo3 = OneClassSVM(kernel='rbf', nu=0.04)

somenteNormais = X_w2v_train[respostasTreino == 0]
modelo3.fit(somenteNormais)

previsoes3 = modelo3.predict(X_w2v_test)

previsoes3_ajustadas = np.where(respostasTeste == 1, -1, 1)

print(classification_report(previsoes3_ajustadas, previsoes3))