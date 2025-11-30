import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Carregar o dataset
df = pd.read_csv('heart.csv')

print("\nPrimeiras linhas do dataset:")
print(df.head())

# Separar os rótulos (target) e os recursos (features)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Identificar colunas categóricas e numéricas
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Configurar pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelos de Machine Learning
models = {
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# Dicionário para armazenar os resultados
results = {}

# Avaliação dos modelos
for model_name, model in models.items():
    print(f"\nTreinando e avaliando {model_name}...")

    # Criar pipeline com pré-processador e modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Treinar o modelo
    pipeline.fit(X_train, y_train)

    # Previsões
    y_pred = pipeline.predict(X_test)

    # Métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy  # Armazenar a acurácia no dicionário
    print(f"Acurácia: {accuracy:.2%}")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

# Comparar os modelos e identificar o melhor
best_model = max(results, key=results.get)
print(f"\nMelhor modelo: {best_model} com acurácia de {results[best_model]:.2%}")