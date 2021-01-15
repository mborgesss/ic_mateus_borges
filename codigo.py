# %%===========================================================================
# BIBLIOTECAS 
# =============================================================================
import pandas as pd
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from spellchecker import SpellChecker
from nltk.stem.snowball import PortugueseStemmer
from unidecode import unidecode
from sklearn.feature_extraction.text import CountVectorizer
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score, classification_report,confusion_matrix
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict, RandomizedSearchCV, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, CategoricalNB, ComplementNB,GaussianNB
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV


# %%===========================================================================
# ABRINDO DADOS 
# =============================================================================
df = pd.read_csv('dados.csv', sep = ';')
df.columns

# %%===========================================================================
# FUNÇÕES 
# =============================================================================

#Excluindo caracteres
def chr_remove(old, to_remove):
    new_string = old
    for x in to_remove:
        new_string = new_string.replace(x, '')
    return new_string

# %%===========================================================================
# PRÉ-PROCESSAMENDO TEXTO
# =============================================================================
#Removendo caracteres especificos do corpus
corpus = df['Relato']

remove=[',','.','/',':','-','_','(',')','@','#', '?','"',
        '0','1','2','3','4','5','6','7','8','9']
corpus=[chr_remove(str(word), remove) for word in corpus]

#Tokenizando (Dividindo a frase em palavras)
tokenizer=TreebankWordTokenizer()
token=[tokenizer.tokenize(lista) for lista in corpus]

# Removendo acento e plural
dic=[]
for i in range(0,len(token)):
    dic.append([stem.stem(unidecode(word)) for word in token[i]])

#Concatenando o resultado
conc = [' '.join(array) for array in dic]
corpus = pd.Series(conc)

# %%===========================================================================
# BAG OF WORDS
# =============================================================================

vectorizer = CountVectorizer(binary = True, ngram_range = (1,3))
X = vectorizer.fit_transform(corpus[:1000])

# %%===========================================================================
# CONJUNTO DE TREINO
# =============================================================================
# causas
causas = dict(enumerate(df.loc[:,'Motivo'].astype('category').cat.categories))

# Variaveis explciativa
X = pd.DataFrame(X.toarray(), columns = vectorizer.get_feature_names()) 

X[['Idade', 'Genero']] = df[['Idade', 'Genero']]

X['Genero'] = X['Genero'].apply(lambda x: 1 if x == 'F' else 0)

# Variável explicada 
Y = np.array(df.loc[:,'Motivo'].astype('category').cat.codes)

# Escalonando
scaler = MinMaxScaler(feature_range = (0,1))
X=scaler.fit_transform(X)

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = .3, 
                                                        random_state = 0)
# %%===========================================================================
# MODELOS
# =============================================================================
knn = KNeighborsClassifier(n_neighbors = 1)
lg = LogisticRegression(penalty = 'l2', random_state = 0, 
                        class_weight = 'balanced', multi_class = 'ovr',
                        solver = 'saga', C = 0.5)

svc = SVC(class_weight = 'balanced', random_state = 0, C = 1,
          gamma = 1, kernel = 'linear')

rf = ExtraTreesClassifier(class_weight = 'balanced',random_state=0,
                          n_estimators = 500, max_features = 'auto',
                          max_depth = 15, min_samples_split = 5,
                          min_samples_leaf = 1, criterion = 'gini')

xgb=XGBClassifier(objective = 'multi:softmax', learning_rate = 0.02, 
                  n_estimators = 300, nthread = 1, 
                  min_child_weight =  1, gamma = 0.5, subsample =  1.0,
                  colsample_bytree =  0.8, max_depth =  5)

nb_Multinomial = MultinomialNB()

nb_Bernoulli = BernoulliNB()

nb_Complement = ComplementNB()

nb_Gaussian = GaussianNB()

# %%===========================================================================
# VALIDAÇÃO CRUZADA
# =============================================================================
modelo=[knn,lg, svc, rf, nb_Multinomial, nb_Bernoulli, 
        nb_Complement, nb_Gaussian, xgb]
resultados = [['Modelo', 'Acurácia', 'Incerteza', 'f1 macro', 'f1 weighted']]
classificadores = ['KNN','Regressão Logistica', 'Support Vector Classifier',
                   'Florestas Aleatórias', 'NB Multinomial','NB Bernoulli', 
                   'NB Complement', 'NB Gaussian', 'XGBClassifier']

for i in range(0, len(modelo)):
    predict = cross_val_predict(estimator = modelo[i], X = X, y = Y, cv = 3)
    acertos = np.array([1 if (predict[x] == Y[x]) else 0 for x in range(0,len(X))])
    resultados.append([classificadores[i], 
                     acertos.mean(), acertos.std()/(1000**0.5),
                     f1_score(Y,predict, average = 'macro'),
                     f1_score(Y,predict, average = 'weighted')])

    print(str(modelo[i]))
    
df_resultados = pd.DataFrame(resultados[1:], columns = resultados[0])