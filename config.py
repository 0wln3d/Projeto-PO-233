# Importa biblioteca para manipulação de caminhos
from pathlib import Path
# Importa biblioteca para manipular arquivos e diretórios
import os

# Importa os algoritmos de classificação do Scikit-Learn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
# from sklearn.neural_network import MLPClassifier  # Comentado

# Importa ferramenta para criar pipeline de pré-processamento e modelo
from sklearn.pipeline import Pipeline

# Importa transformações para pré-processamento dos dados
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA

# Importa validação cruzada e busca em grade
from sklearn.model_selection import GridSearchCV, StratifiedKFold
# Importa métricas de avaliação
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import make_scorer

# Semente aleatória para reprodutibilidade
seed = 352163

# Cria diretório para salvar resultados
results_dir = Path('./results')
results_dir.mkdir(parents=True, exist_ok=True)

# Define diretório onde estão os arquivos .csv
db_dir = './db/'

# Função que carrega todos os arquivos .csv do diretório informado
def get_csv_files_dict(path=db_dir):
    files = {}
    for file in os.listdir(path):
        if file.endswith('.csv'):
            files[file] = os.path.join(path, file)
    return files

# Define qual métrica será usada como base para otimização
scorer = make_scorer(accuracy_score)

# Validação cruzada: divisão estratificada (mantém proporção das classes)
cv = StratifiedKFold(n_splits=10, shuffle=False)
gscv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

# Define dicionário com os algoritmos e seus respectivos pipelines e grids de hiperparâmetros
algorithms = {
    'kNN': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),            # Imputação de valores ausentes
            ('scaler', MinMaxScaler(feature_range=(0, 1))),         # Normalização
            ('pca', PCA(n_components=0.95, random_state=seed)),     # Redução de dimensionalidade
            ('knn', KNeighborsClassifier())                         # Algoritmo kNN
        ]),
        param_grid={
            'pca__n_components': [0.90, 0.95, 0.99],                # Quantidade de variância explicada no PCA
            'knn__n_neighbors': [2, 3, 4, 6, 8],                    # Número de vizinhos
            'knn__p': [1, 2],                                       # Distância de Manhattan (1) ou Euclidiana (2)
        },
        scoring=scorer,
        cv=gscv
    ),

    'tree': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('pca', PCA(n_components=0.95, random_state=seed)),
            ('tree', DecisionTreeClassifier(random_state=seed))     # Árvore de decisão
        ]),
        param_grid={
            'pca__n_components': [0.90, 0.95, 0.99],
            'tree__max_depth': [5, 10, 20],                         # Profundidade da árvore
            'tree__criterion': ['entropy', 'gini'],                 # Critério de divisão
        },
        scoring=scorer,
        cv=gscv
    ),

    'bigtree': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('pca', PCA(n_components=0.95, random_state=seed)),
            ('tree', DecisionTreeClassifier(max_depth=None, min_samples_split=2, random_state=seed))
        ]),
        param_grid={
            'pca__n_components': [0.90, 0.95, 0.99],
            'tree__criterion': ['entropy', 'gini'],
        },
        scoring=scorer,
        cv=gscv
    ),

    'nb': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),                           # Padronização
            ('feature_selection', SelectKBest(score_func=mutual_info_classif)),  # Seleção de atributos
            ('nb', GaussianNB())                                    # Naive Bayes
        ]),
        param_grid={
            'feature_selection__k': [2, 4, 6, 8],                   # Número de atributos selecionados
        },
        scoring=scorer,
        cv=gscv
    ),

    'svmlinear': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=seed)),
            ('svm', SVC(kernel='linear', random_state=seed))        # SVM linear
        ]),
        param_grid={
            'pca__n_components': [0.90, 0.95, 0.99],
            'svm__C': [0.5, 1.0, 2.0],                               # Parâmetro de regularização
        },
        scoring=scorer,
        cv=gscv
    ),

    'svmrbf': GridSearchCV(
        Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=0.95, random_state=seed)),
            ('svm', SVC(kernel='rbf', random_state=seed))           # SVM com kernel radial
        ]),
        param_grid={
            'pca__n_components': [0.90, 0.95, 0.99],
            'svm__C': [0.5, 1.0, 2.0],
            'svm__gamma': [0.01, 0.1, 1.0],                         # Parâmetro gamma do kernel
        },
        scoring=scorer,
        cv=gscv
    ),
}
