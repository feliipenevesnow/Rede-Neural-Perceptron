import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import openpyxl
import time

# 3) Função de ativação: degrau;
def ativacao(soma):
    if soma >= 0:
        return 1
    return 0

df = pd.read_excel('Banco de dados para a RNA Perceptron.xlsx')

dados = df.loc[3:192, df.columns[2:5]]
dados_array = dados.to_numpy()
dados_desligar = [[dados_array[i, j] for j in range(dados_array.shape[1])] for i in range(dados_array.shape[0])]

dados = df.loc[3:198, df.columns[6:11]]
dados_array = dados.to_numpy()
dados_ligar = [[dados_array[i, j] for j in range(dados_array.shape[1])] for i in range(dados_array.shape[0])]

# Mescla os dados
dados = np.concatenate((dados_ligar, dados_desligar), axis=0)

# Divide os dados em conjuntos de treinamento e teste
np.random.shuffle(dados)
porcentagem_treinamento = 0.8
quantidade_treinamento = int(porcentagem_treinamento * len(dados))
dados_treinamento = dados[:quantidade_treinamento]
dados_teste = dados[quantidade_treinamento:]

# 1) Valores iniciais dos pesos e bias: definir pseudo-aleatoriamente entre 0 e 1
# Bias, w1, w2, taxa
bias = random.uniform(0, 1)
w1 = random.uniform(0, 1)
w2 = random.uniform(0, 1)

# 2) Taxa de aprendizagem: definida pelo usuário (para definir a taxa de aprendizagem reflita sobre a
# questão da estabilidade da rede neural artificial)
taxa = 0.5


erro = 1
epocas = 0

inicio_treinamento = time.perf_counter()
while erro > 0:
    erro = 0
    for dado in dados_treinamento:
        soma = (dado[0] * w1) + (dado[1] * w2) + (-1 * bias)
        resultado = ativacao(soma)
        if dado[2] != resultado:
            erro = 1
            w1_anterior = w1
            w1 = w1_anterior + taxa * (dado[2] - resultado) * dado[0]
            w2_anterior = w2
            w2 = w2_anterior + taxa * (dado[2] - resultado) * dado[1]
            bias_anterior = bias
            bias = bias_anterior + taxa * (dado[2] - resultado) * -1
    epocas = epocas + 1
fim_treinamento = time.perf_counter()

confusao = [[0, 0], [0, 0]]

inicio_teste = time.perf_counter()
for dado in dados_teste:
        soma = dado[0] * w1 + dado[1] * w2 + -1 * bias
        resultado = ativacao(soma)
        if dado[2] != resultado:
            if dado[2] == 1:
                confusao[0][1] = confusao[0][1] + 1
            else:
                confusao[1][0] = confusao[1][0] + 1
        else:
            if dado[2] == 1:
                confusao[1][1] = confusao[1][1] + 1
            else:
                confusao[0][0] = confusao[0][0] + 1
fim_teste = time.perf_counter()

# Gerando pontos
x = []
y = []
cores = []
for dado in dados_treinamento:
    x.append(dado[0])
    y.append(dado[1])
    if dado[2] == 0:
       cores.append('r')
    else:
       cores.append('b')


plt.scatter(x, y, color=cores)

x = range(-1, 2)

y = [((-w1)/w2 * i) + bias/w2 for i in x]

plt.plot(x, y)

plt.title("Treinamento")

plt.show()

# Gerando pontos
x = []
y = []
cores = []
for dado in dados_teste:
    x.append(dado[0])
    y.append(dado[1])
    if dado[2] == 0:
       cores.append('r')
    else:
       cores.append('b')


plt.scatter(x, y, color=cores)

x = range(-1, 2)

y = [((-w1)/w2 * i) + bias/w2 for i in x]

plt.plot(x, y)

plt.title("Teste")

plt.show()




duracao = fim_treinamento - inicio_treinamento # duração em segundos
print(f"O treinamento demorou {duracao:.6f} segundos para ser executado.")

duracao = fim_teste - inicio_teste # duração em segundos
print(f"O teste demorou {duracao:.6f} segundos para ser executado.")

#total de epocas
print("Epocas: " + str(epocas))

#Percentual de acertos na etapa de teste
soma = confusao[0][1] + confusao[1][0]
perc = (soma * 100) / len(dados_teste)
print("Percentual de acertos: " + str(100 - perc) + "%")

#Matriz confusao
print("Matriz confusão\n  0   1")
print("0" + str(confusao[0]))
print("1" + str(confusao[1]))