#Aprendizado Não Supervisionado: 
#No aprendizado não supervisionado, o modelo não é fornecido com rótulos ou classes para os dados de treinamento.
#Em vez disso, o objetivo é encontrar estruturas e padrões nos dados sem orientação externa. O exemplo fornecido usa o algoritmo K-Means para agrupar as
#descrições de animais em clusters com base na similaridade dos dados, mas não é fornecido um rótulo específico para cada cluster. O algoritmo agrupa os dados
#com base em características intrínsecas e semelhanças,
#sem conhecimento prévio das categorias ou classes.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np

#Nestas linhas, estamos importando as bibliotecas necessárias para o código. TfidfVectorizer é uma
#classe que ajuda a vetorizar texto usando a ponderação TF-IDF,
#KMeans é um algoritmo de agrupamento, e numpy é uma biblioteca amplamente usada para operações matriciais.

# Descrições de animais
descriptions = [
    "Um animal de estimação com pelos macios e que ronrona.",
    "Um animal leal e amigável que adora brincar.",
    "Uma ave pequena e colorida que canta lindamente.",
    "Um animal selvagem com listras e presas afiadas.",
    "Uma espécie de réptil que rasteja e tem escamas.",
    "Um mamífero marinho gigante que vive nos oceanos.",
    "Uma ave de rapina que tem garras afiadas e voa alto.",
    "Um mamífero pequeno e noturno que dorme de cabeça para baixo.",
    "Um animal de fazenda que produz ovos e carne.",
    "Uma ave com penas coloridas e um bico longo.",
    "Um animal de estimação peludo que gosta de brincar com bolas.",
    "Um mamífero selvagem com uma juba majestosa.",
    "Um réptil que rasteja e pode ser venenoso.",
    "Uma espécie de marsupial que vive na Austrália.",
    "Um grande herbívoro com chifres.",
    "Uma ave que migra sazonalmente em longas distâncias.",
    "Um animal marinho com barbatanas e cauda bifurcada.",
    "Um pequeno roedor que gosta de cavar buracos.",
    "Uma ave noturna que emite sons misteriosos à noite.",
    "Um grande felino selvagem com manchas.",
    "Um réptil que rasteja e é conhecido por sua casca dura.",
    "Uma espécie de primata conhecida por sua cauda preênsil.",
    "Um grande mamífero herbívoro da savana africana.",
    "Uma ave que não voa e é conhecida por sua velocidade.",
    "Um pequeno roedor com cauda longa e peluda.",
    "Uma ave aquática com um longo pescoço.",
    "Um felino doméstico que gosta de brincar com bolas de lã.",
    "Um animal noturno com olhos grandes e orelhas pontiagudas.",
    "Um mamífero que vive nas árvores e se move lentamente.",
    "Uma ave tropical de cores vibrantes.",
    "Um mamífero marinho que realiza saltos espetaculares na água."
]
#Aqui, definimos uma lista chamada descriptions, que contém descrições de diferentes animais.

# Vetorização das descrições com TF-IDF
vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(descriptions)

#Nestas linhas, criamos um objeto vectorizer do tipo TfidfVectorizer que irá ajudar a vetorizar as
#descrições usando a ponderação TF-IDF. O TF-IDF (Term Frequency-Inverse Document Frequency) é uma
#técnica que converte texto em uma representação numérica.
#X é a matriz resultante após a vetorização, onde cada linha representa uma descrição
#e cada coluna representa um termo no texto.

# Agrupamento com K-Means
num_clusters = 5  # Defina o número de clusters desejado
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(X)

#Nestas linhas, definimos o número de clusters desejado (neste caso, 5) e, em seguida,
#criamos um objeto kmeans do tipo KMeans com esse número de clusters.
#O algoritmo K-Means será usado para agrupar as descrições de animais.
#O método fit_predict ajusta o modelo K-Means aos dados vetorizados e atribui cada descrição a um cluster.

# Exibir as descrições e os clusters aos quais pertencem
for descricao, cluster in zip(descriptions, clusters):
    print(f"Descrição: {descricao} - Cluster: {cluster}")

#Nesta parte, estamos percorrendo as descrições e os clusters atribuídos a elas e imprimindo a descrição juntamente com o cluster ao qual pertencem.

#Em resumo, este código usa a vetorização de texto com TF-IDF e o algoritmo K-Means para agrupar as descrições de animais
#em clusters com base em sua similaridade textual. A saída final mostra as descrições e os clusters aos quais pertencem.
#Isso pode ser útil para agrupar automaticamente descrições de animais semelhantes com base em seu conteúdo textual.