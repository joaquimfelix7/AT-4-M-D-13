import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import requests

# Projeto 1: Pipeline com Scikit-Learn para prever preço de automóveis

def car_price_pipeline():
    # Simulando um dataset
    data = pd.DataFrame({
        'marca': ['Ford', 'Chevrolet', 'Toyota', 'Honda', 'Ford', 'Chevrolet'],
        'ano': [2015, 2018, 2016, 2017, 2014, 2019],
        'quilometragem': [50000, 30000, 60000, 45000, 70000, 25000],
        'preco': [45000, 55000, 48000, 50000, 42000, 60000]
    })

    X = data.drop(columns=['preco'])
    y = data['preco']

    cat_features = ['marca']
    num_features = ['ano', 'quilometragem']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(), cat_features),
        ('num', StandardScaler(), num_features)
    ])

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Erro Quadrático Médio: {mse}')

# Projeto 2: Previsão de preços de criptomoedas usando API

def crypto_price_prediction():
    API_URL = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=10&interval=daily"
    response = requests.get(API_URL)
    data = response.json()
    
    prices = [entry[1] for entry in data['prices']]
    df = pd.DataFrame({'preco': prices})
    df['dia_anterior'] = df['preco'].shift(1)
    df.dropna(inplace=True)
    df['subiu'] = (df['preco'] > df['dia_anterior']).astype(int)
    
    X = df[['dia_anterior']]
    y = df['subiu']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Acurácia: {accuracy}')

if __name__ == "__main__":
    car_price_pipeline()
    crypto_price_prediction()
