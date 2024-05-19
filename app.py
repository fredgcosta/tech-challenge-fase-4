# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px

import tensorflow as tf
from tensorflow.keras.models import load_model
keras = tf.keras

LOGGER = get_logger(__name__)



@st.cache_resource
def load_lstm_model():
    lstm_model = load_model("models/lstm_model.keras", safe_mode=False)
    lstm_model.make_predict_function()
    lstm_model.summary()
    return lstm_model
@st.cache_data
def get_data():
    df = pd.read_csv('Europe_Brent_Spot_Price_FOB.csv', parse_dates=True, index_col=0)
    df = df.sort_values(by='Date', ascending=True)
    df['Log Returns'] = np.log(df['Close']/df['Close'].shift(1)).dropna()
    return df

def standardize_data(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df

def unstandardize_data(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df =   (train_df * train_std) + train_mean
    val_df = (val_df * train_std) + train_mean
    test_df = (test_df * train_std) + train_mean

    return train_df, val_df, test_df

def standardize_data2(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std

    return train_df, val_df, test_df

def unstandardize_data2(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df =   (train_df * train_std) + train_mean
    val_df = (val_df * train_std) + train_mean
    test_df = (test_df * train_std) + train_mean

    return train_df, val_df, test_df


def split_data(df):
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n*0.7)]
    val_df = df[int(n*0.7):int(n*0.9)]
    test_df = df[int(n*0.9):]

    num_features = df.shape[1]
    return train_df, val_df, test_df

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    return plt

def run():

    
    st.set_page_config(
        page_title="Tech Challenge - Fase 4",
        page_icon=":chart_with_upwards_trend:",
        layout="wide"
    )
    st.title(":chart_with_upwards_trend: Previsão do preço do Petróleo Brent")

    st.write("## Tech Challenge - Fase 4 ")
    
    tab0, tab1, tab2, tab3, tab4 = st.tabs(['Introdução', 'Insights', 'Modelo', 'Código Modelo', 'Conclusão'])


    with tab0:

        st.markdown(
            """
    ### Introdução

    Neste projeto iremos analisar a série temporal do petróleo Brent, entendendo os fatores que impactam na variação de seu valor, e por fim, propor um modelo de Machine Learning para fazer a previsão de valores futuros utilizando as bibliotecas TensorFlow e Keras.
    
    Membros do grupo 24:

    - RM351388 - Carolina Pasianot Casetta - carol_pasianot@hotmail.com
    - RM351418 - Gustavo França Severino - gustavofs.dt@gmail.com
    - RM352372 - Frederico Garcia Costa - fredgcosta@gmail.com
    - RM351905 - Jeferson Vieira - jvieirax@gmail.com
    - RM351187 - Victor Wilson Costa Lamana - victor.lamana15@gmail.com

    
        """
        )

    with tab2:

        st.markdown("""Para a reliazação do treinamento do modelo foi considerado o período de 1987 à 2012.""")
        st.markdown("""A seguir temos o teste de validação do modelo onde utilizamos o período de 2013 à 2020 para tal verificação.""")

        model = load_lstm_model()

        df = get_data()

        df = df.drop(columns=['Log Returns']) # Removendo colunas desnecessárias

        train_df, val_df, test_df = split_data(df)
        train_df, val_df, test_df = standardize_data(train_df, val_df, test_df)

        val_array = np.array(val_df['Close'], dtype=np.float32)
        val_rnn_forecast = model.predict(val_array[np.newaxis, :, np.newaxis])
        val_rnn_forecast = val_rnn_forecast[0, :len(val_array), 0]

        import plotly.graph_objs as go     

        fig = px.line(val_df, x=val_df.index, y='Close', title='Série temporal de fechamento')
        fig.update_xaxes(title_text='Data')
        fig.update_yaxes(title_text='Cotação')
        fig.add_trace(go.Scatter(x=val_df.index, y=val_rnn_forecast, mode='lines', name='Previsão RNN'))

        fig.update_layout(width=1200, height=700)

        st.plotly_chart(fig)

    with tab1:

        st.markdown(""" Com o boxplot abaixo, é possível analisar a variância do preço do petróleo ao longo dos anos""")
        df['Year'] = df.index.year

        fig = px.box(df, x='Year', y='Close', color='Year', title='Variação anual do petróleo Brent de 1987 a 2024')
        fig.update_xaxes(title_text='Data', tickangle=60, tickmode='linear', dtick =1)
        fig.update_yaxes(title_text='Cotação')
        fig.add_shape(dict(type='line', x0='1990', x1='1990', y0=df['Close'].min(), y1=df['Close'].max(), line=dict(color='lightblue')))
        fig.add_shape(dict(type='line', x0='2008', x1='2008', y0=df['Close'].min(), y1=df['Close'].max(), line=dict(color='lightblue')))
        fig.add_shape(dict(type='line', x0='2014', x1='2014', y0=df['Close'].min(), y1=df['Close'].max(), line=dict(color='lightblue')))
        fig.add_shape(dict(type='line', x0='2020', x1='2020', y0=df['Close'].min(), y1=df['Close'].max(), line=dict(color='lightblue')))
        fig.add_shape(dict(type='line', x0='2022', x1='2022', y0=df['Close'].min(), y1=df['Close'].max(), line=dict(color='lightblue')))

        fig.update_layout(showlegend=False, template='plotly_white', width=1000)

        fig.update_layout(showlegend=False, template='plotly_white')


        st.plotly_chart(fig)
        
        st.markdown("""
                    Podemos observar uma volatilidade anormal principalmente nos anos de 1990, 2008, 2014, 2020 e 2022. As principais razões para essas anomalias estão listadas abaixo:

                    1990: Guerra do Golfo e inverno rigoroso.

                    2008: Crise financeira (A Grande Recessão).

                    2014: Preocupações geopolíticas reduzidas e oferta constante de petróleo pela OPEC.

                    2020: Pandemia de Covid-19 e restrições de viagem.

                    2022: Guerra entre Rússia e Ucrânia.

                    Esses eventos mencionados acima são frequentemente considerados anomalias, tornando a série altamente volátil e imprevisível, já que a maioria dos modelos assumem que os dados são homocedásticos (ou seja, a média e a variância permanecem constantes ao longo do tempo).
                    """)
        
    with tab3:

        st.markdown("""Abaixo segue o código do modelo LSTM utilizado para fazer a previsão:""")

        codigo_python = """
            # Definir o dispositivo para a GPU se disponível
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Carregar o DataFrame
            df = pd.read_csv('/content/ipea.csv')

            # Converter a coluna de data para datetime e depois para timestamp Unix
            df['Data'] = pd.to_datetime(df['Data'], dayfirst=True)
            df = df.sort_values(by='Data',ascending=True)
            df['Timestamp'] = df['Data'].values.astype('int64') // 10**9

            # Escalar a coluna de preços, já que os modelos de DL geralmente funcionam melhor com dados normalizados
            scaler = MinMaxScaler(feature_range=(-1, 1))
            df['Preço - petróleo bruto - Brent (FOB)'] = scaler.fit_transform(df['Preço - petróleo bruto - Brent (FOB)'].values.reshape(-1, 1)).astype('float32')

            # Preparar dados para o PyTorch
            X = df['Timestamp'].values.astype('float32')  # A entrada do modelo será o timestamp
            y = df['Preço - petróleo bruto - Brent (FOB)'].values.astype('float32')  # A saída do modelo serão os preços

            # Dividir o conjunto de dados em treinamento e teste
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # Converter os dados para Tensor
            X_train_tensor = torch.tensor(X_train).view(-1, 1, 1)
            y_train_tensor = torch.tensor(y_train).view(-1, 1, 1)
            X_test_tensor = torch.tensor(X_test).view(-1, 1, 1)
            y_test_tensor = torch.tensor(y_test).view(-1, 1, 1)

            # Mover para o dispositivo apropriado
            X_train_tensor = X_train_tensor.to(device)
            y_train_tensor = y_train_tensor.to(device)
            X_test_tensor = X_test_tensor.to(device)
            y_test_tensor = y_test_tensor.to(device)

            # Definir o modelo LSTM
            class LSTMModel(nn.Module):
                def __init__(self, input_size=1, hidden_layer_size=200, output_size=1):
                    super(LSTMModel, self).__init__()
                    self.hidden_layer_size = hidden_layer_size

                    self.lstm = nn.LSTM(input_size, hidden_layer_size ,num_layers=3)

                    self.linear = nn.Linear(hidden_layer_size, output_size)

                def forward(self, input_seq):
                    lstm_out, _ = self.lstm(input_seq)
                    predictions = self.linear(lstm_out.view(len(input_seq), -1))
                    return predictions[-1]

            # Instanciar o modelo
            model = LSTMModel().to(device)

            # Definir a função de perda e o otimizador
            loss_function = nn.MSELoss()
            optimizer = Adam(model.parameters(), lr=0.0001)

            # Treinar o modelo
            epochs = 10
            for i in range(epochs):
                for seq, labels in zip(X_train_tensor, y_train_tensor):
                    optimizer.zero_grad()

                    model.hidden = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                                    torch.zeros(1, 1, model.hidden_layer_size).to(device))

                    y_pred = model(seq)

                    single_loss = loss_function(y_pred, labels)
                    single_loss.backward()
                    optimizer.step()

                    print(f'Epoch {i} loss: {single_loss.item()}')

            model.eval()
            with torch.no_grad():
                preds = []
                for i in range(len(X_test)):
                    seq = X_test_tensor[i : i + 1]
                    preds.append(model(seq).cpu().numpy()[0])

            # Inverter a escala dos preços para a escala original
            actual_predictions = scaler.inverse_transform(np.array(preds).reshape(-1, 1))
        """
        st.code(codigo_python, language='python')

    with tab4:

        st.markdown("""Abaixo temos a previsão realizada pelo modelo do período que vai de 2021 à 2024. Como já temos conhecimento dos valores de fechamento desse período é possível compará-los aos valores previstos pelo modelo.""")

        model = load_lstm_model()

        df = get_data()

        df = df.drop(columns=['Log Returns']) # Removendo colunas desnecessárias

        train_df, val_df, test_df = split_data(df)
        train_df1, val_df1, test_df1 = split_data(df)
        train_df, val_df, test_df = standardize_data2(train_df, val_df, test_df)

        val_array = np.array(test_df['Close'], dtype=np.float32)
        previsao = model.predict(val_array[np.newaxis, :, np.newaxis])
        previsao = previsao[0, :len(val_array), 0]

        import plotly.graph_objs as go 

        novoDF = pd.DataFrame(test_df)
        novoDF['Close'] = previsao

        train_df1, val_df1, novoDF = unstandardize_data2(train_df1, val_df1, novoDF)

        fig = px.line(test_df1, x=test_df1.index, y='Close', title='Série temporal de fechamento')
        fig.update_xaxes(title_text='Data')
        fig.update_yaxes(title_text='Cotação')
        fig.add_trace(go.Scatter(x=test_df1.index, y=novoDF['Close'], mode='lines', name='Previsão RNN'))

        fig.update_layout(width=1200, height=700)

        st.plotly_chart(fig)

        st.markdown("""É válido lembrar, como foi citado na aba de insights, que eventos de grande magnitude podem causar um impacto significativo na economia global e consequentemente na variação do preço do Petróleo, dificultando a previsão e diminuindo a acertividade do modelo.""")

if __name__ == "__main__":
    run()
#%%
