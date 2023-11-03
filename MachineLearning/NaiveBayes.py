#Aprendizado Supervisionado: 
#No aprendizado supervisionado, um modelo é treinado com um conjunto de dados rotulados,
#o que significa que cada exemplo no conjunto de treinamento é associado a um rótulo ou uma classe conhecida. O modelo
#aprende a mapear os recursos dos dados para os rótulos correspondentes,
#tornando-se capaz de fazer previsões ou classificar novos exemplos com base nesse aprendizado.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Descrições de animais
#Aqui são fornecidas descrições de animais em forma de uma lista de strings.
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

# Rótulos correspondentes
#São fornecidos os rótulos correspondentes para as descrições de animais.
labels = [
    "gato", "cachorro", "passarinho", "tigre", "cobra", "baleia", "águia",
    "morcego", "galinha", "papagaio", "hamster", "leão", "lagarto", "canguru",
    "bisão", "andorinha", "golfinho", "esquilo", "coruja", "leopardo", "tartaruga",
    "macaco", "elefante", "avestruz", "rato", "corça", "jaguar", "preguiça",
    "tucano", "golfinho", "desconhecido"
]

# Vetorização das descrições com TF-IDF
#Nesta parte, as descrições são convertidas em vetores TF-IDF (Term Frequency-Inverse Document Frequency) 
#usando o TfidfVectorizer. O parâmetro lowercase=True indica que as palavras devem ser convertidas para minúsculas
#antes da vetorização.
vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(descriptions)

# Treinamento do modelo com Multinomial Naive Bayes
#Aqui, um modelo de classificação Naive Bayes Multinomial é treinado com base nos vetores TF-IDF e rótulos correspondentes.
#O valor de alpha é um hiperparâmetro do modelo, que você pode ajustar conforme necessário.
clf = MultinomialNB(alpha=0.1)  # Ajuste o valor de alpha conforme necessário
clf.fit(X, labels)

# Previsão
#Uma nova descrição (no caso, "roedor com cauda longa") é fornecida, e ela é transformada em um vetor TF-IDF usando o mesmo vectorizer.
#Em seguida, o modelo prevê a categoria da nova descrição.
nova_descricao = ["rasteja e casca bem dura"]
#    POSSIVEIS DESCRIÇÕES:
#    "Um animal aquático com nadadeiras e escamas.",
#    "Um inseto pequeno que zumbi no verão.",
#    "Uma criatura voadora com penas coloridas e bico longo.",
#    "Um mamífero marinho com barbatanas e dentes afiados.",
#    "Um roedor com cauda longa e dentes incisivos.",
#    "Uma ave noturna com olhos grandes e penas suaves.",
#    "Um réptil que rasteja e é conhecido por mudar de cor.",
#    "Uma espécie de primata que vive nas florestas tropicais.",
#    "Um animal terrestre com uma concha protetora.",
#    "Um grande felino selvagem com garras afiadas.",

X_nova = vectorizer.transform(nova_descricao)
categoria_prevista = clf.predict(X_nova)

# Exibição da categoria prevista
print("A categoria prevista para a nova descrição é:", categoria_prevista[0])


#O "Naive Bayes" é um algoritmo de aprendizado de máquina com supervisão que é comumente usado para classificação e análise de texto.
#Ele é baseado no Teorema de Bayes e é "ingênuo" porque assume independência condicional entre os recursos (palavras ou características) usados para a classificação,
#mesmo que essa independência nem sempre seja verdadeira na prática.
#Devido a essa suposição, o Naive Bayes pode não ser o melhor algoritmo para todos os tipos de dados, mas é amplamente utilizado em aplicações de processamento de linguagem
#natural e classificação de textos, como detecção de spam, classificação de documentos, análise de sentimentos e muito mais.

#Em resumo, o Naive Bayes é um algoritmo de classificação que calcula a probabilidade de um ponto de dados pertencer a uma classe com base em características (ou palavras)
#e a suposição de independência condicional entre essas características. É uma técnica eficaz e rápida para tarefas de classificação de texto, mas pode não funcionar bem em todos os cenários,
#especialmente quando a independência condicional não é uma suposição razoável.

#O projeto que você descreveu se encaixa no Aprendizado Supervisionado e utiliza um algoritmo de Naive Bayes para classificar descrições de animais em categorias predefinidas com base em rótulos fornecidos.
#O aprendizado supervisionado envolve o treinamento de um modelo com um conjunto de dados que contém pares de entrada e saída (neste caso, descrições de animais e rótulos correspondentes). 
#O modelo é treinado para fazer previsões com base nesses exemplos de treinamento.

#Nesse projeto, as descrições de animais são as entradas (recursos) e as categorias (por exemplo, "gato", "cachorro", etc.) são as saídas que o modelo tenta prever.
#O modelo é treinado usando o algoritmo de classificação Naive Bayes Multinomial com vetores TF-IDF das descrições como características. Após o treinamento,
#o modelo é capaz de prever a categoria de uma nova descrição de animal com base no que aprendeu durante o treinamento.

#Portanto, seu projeto está relacionado ao Aprendizado Supervisionado e usa o Aprendizado de Máquina para classificar dados com base em rótulos fornecidos durante o treinamento.