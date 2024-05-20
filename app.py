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
import yfinance as yf

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
    return df

@st.cache_data
def get_new_data():
    df = get_data()

    yf.pdr_override()
    yf.set_tz_cache_location(".cache/py-yfinance")

    # Especifique o símbolo do petróleo Brent (BZ=F) e o intervalo de datas desejado
    symbol = 'BZ=F'
    start_date = '2024-05-14'

    # Use a função download para obter os dados
    yf_brent = yf.download(symbol, start=start_date)
    yf_brent = yf_brent.drop(columns=['Open', 'High', 'Low', 'Volume', 'Adj Close']) # Removendo colunas desnecessárias
    yf_brent = yf_brent.sort_values(by='Date', ascending=True)
    yf_brent.info()
    df = pd.concat([df,yf_brent])

    df['Log Returns'] = np.log(df['Close']/df['Close'].shift(1)).dropna()
    return df

def standardize_data(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    s_train_df = (train_df - train_mean) / train_std
    s_val_df = (val_df - train_mean) / train_std
    s_test_df = (test_df - train_mean) / train_std

    return s_train_df, s_val_df, s_test_df

def unstandardize_data(train_df, s_train_df, s_val_df, s_test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (s_train_df * train_std) + train_mean
    val_df = (s_val_df * train_std) + train_mean
    test_df = (s_test_df * train_std) + train_mean

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

        st.markdown("""Para a realização do treinamento do modelo foi considerado o período de 1987 à 2012.""")
        st.markdown("""A seguir temos o teste de validação do modelo onde utilizamos o período de 2013 à 2020 para tal verificação.""")

        model = load_lstm_model()

        df = get_new_data()

        df = df.drop(columns=['Log Returns']) # Removendo colunas desnecessárias

        train_df, val_df, test_df = split_data(df)
        s_train_df, s_val_df, s_test_df = standardize_data(train_df, val_df, test_df)

        val_array = np.array(s_val_df['Close'], dtype=np.float32)
        val_rnn_forecast = model.predict(val_array[np.newaxis, :, np.newaxis])
        val_rnn_forecast = val_rnn_forecast[0, :len(val_array), 0]

        novoDF = pd.DataFrame(val_df)
        novoDF['Close'] = val_rnn_forecast

        train_df, val_df, novoDF = unstandardize_data(train_df, s_train_df, s_val_df, novoDF)

        import plotly.graph_objs as go     

        fig = px.line(val_df, x=val_df.index, y='Close', title='Série temporal de fechamento')
        fig.update_xaxes(title_text='Data')
        fig.update_yaxes(title_text='Cotação')
        fig.add_trace(go.Scatter(x=novoDF.index, y=novoDF['Close'], mode='lines', name='Previsão RNN'))

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
            import os
            import IPython.display
            import matplotlib as mpl
            import matplotlib.pyplot as plt
            import numpy as np
            import pandas as pd
            import seaborn as sns
            import tensorflow as tf
            keras = tf.keras
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')
            
            %matplotlib inline
            mpl.rcParams['figure.figsize'] = (20, 6)
            mpl.rcParams['axes.grid'] = False
            mpl.rcParams['font.size'] = 12
            
            print("Quantidade de GPUs disponíveis: ", len(tf.config.list_physical_devices('GPU')))
            
            brent = pd.read_csv('Europe_Brent_Spot_Price_FOB.csv', parse_dates=True, index_col=0)
            brent = brent.sort_values(by='Date', ascending=True)
            brent.info()
            
            df = brent.copy(deep=True)
            df = df.drop(columns=['Log Returns']) # Removendo colunas desnecessárias
      
            # Separando os dados em treino, validação e teste
            
            column_indices = {name: i for i, name in enumerate(df.columns)}

            n = len(df)
            train_df = df[0:int(n*0.7)]
            val_df = df[int(n*0.7):int(n*0.9)]
            test_df = df[int(n*0.9):]
            
            num_features = df.shape[1]
            
            # Normalizando os dados usando a fórmula (df - df.mean()) / df.std()
            
            train_mean = train_df.mean()
            train_std = train_df.std()
            
            train_df = (train_df - train_mean) / train_std
            val_df = (val_df - train_mean) / train_std
            test_df = (test_df - train_mean) / train_std
            
            len_train_df = len(train_df)
            len_val_df = len(val_df)
            len_test_df = len(test_df)
            
            # Funções para plotar a previsão e criar os datasets de janelas sequenciais
            
            def plot_series(time, series, format="-", start=0, end=None, label=None):
                plt.plot(time[start:end], series[start:end], format, label=label)
                plt.xlabel("Time")
                plt.ylabel("Value")
                if label:
                    plt.legend(fontsize=14)
                plt.grid(True)
            
            def sequential_window_dataset(series, window_size):
                series = tf.expand_dims(series, axis=-1)
                ds = tf.data.Dataset.from_tensor_slices(series)
                ds = ds.window(window_size + 1, shift=window_size, drop_remainder=True)
                ds = ds.flat_map(lambda window: window.batch(window_size + 1))
                ds = ds.map(lambda window: (window[:-1], window[1:]))
                return ds.batch(1).prefetch(1)
                
            # preparando os dados no formato de numpy array
                
            x_train = np.array(train_df['Close'], dtype=np.float32)
            x_train = tf.constant(x_train)
            x_valid = np.array(val_df['Close'], dtype=np.float32)
            x_valid = tf.constant(x_valid)
            x_test = np.array(test_df['Close'], dtype=np.float32)
            x_test = tf.constant(x_test)
            
            # Função de Callback do Keras para resetar o estado do modelo a cada epoch
            
            class ResetStatesCallback(keras.callbacks.Callback):
                def on_epoch_begin(self, epoch, logs):
                    if(model.name != 'sequential'):
                        self.model.reset_states()
            
            # Encontrando o melhor valor para a taxa de aprendizado learning_rate do otimizador do modelo
            
            keras.backend.clear_session()
            tf.random.set_seed(42)
            np.random.seed(42)
            
            window_size = 30
            train_set = sequential_window_dataset(x_train, window_size)
            
            model = keras.models.Sequential([
                keras.Input(batch_shape=[1, None, 1]),
                keras.layers.LSTM(100, return_sequences=True, stateful=True),
                keras.layers.LSTM(100, return_sequences=True, stateful=True),
                keras.layers.Dense(1),
                keras.layers.Lambda(lambda x: x * 200.0)
            ])
            lr_schedule = keras.callbacks.LearningRateScheduler(lambda epoch: 1e-9 * 10**(epoch / 20))
            reset_states = ResetStatesCallback()
            optimizer = keras.optimizers.SGD(learning_rate=1e-9, momentum=0.9)
            model.compile(loss=keras.losses.Huber(),
                          optimizer=optimizer,
                          metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.MeanSquaredError()])
            history = model.fit(train_set, epochs=100, callbacks=[lr_schedule, reset_states])
        
            # Plotando o gráfico entre loss e learning_rate
            
            plt.semilogx(history.history["learning_rate"], history.history["loss"])
            plt.axis([1e-9, 1e-4, 0, 10])
            
            #Encontramos o learning_rate ideal como sendo 5e-7, ou seja, 0.0000005
            
            # Treinamos novamente o modelo, usando o learning_rate = 5e-7 e agora usando os dados de validação para validar o modelo
            
            keras.backend.clear_session()
            tf.random.set_seed(42)
            np.random.seed(42)
            
            window_size = 30
            train_set = sequential_window_dataset(x_train, window_size)
            valid_set = sequential_window_dataset(x_valid, window_size)
            
            model = keras.models.Sequential([
                keras.Input(batch_shape=[1, None, 1]),
                keras.layers.LSTM(100, return_sequences=True, stateful=True),
                keras.layers.LSTM(100, return_sequences=True, stateful=True),
                keras.layers.Dense(1),
                keras.layers.Lambda(lambda x: x * 200.0)
            ])
            optimizer = keras.optimizers.SGD(learning_rate=5e-7, momentum=0.9)
            model.compile(loss=keras.losses.Huber(),
                          optimizer=optimizer,
                          metrics=[keras.metrics.MeanAbsoluteError(), keras.metrics.MeanSquaredError()])
            reset_states = ResetStatesCallback()
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                "lstm_model.keras", save_best_only=True)
            early_stopping = keras.callbacks.EarlyStopping(patience=50)
            model.fit(train_set, epochs=500,
                      validation_data=valid_set,
                      callbacks=[early_stopping, model_checkpoint, reset_states])
            
            #Salvamos o melhor modelo e em seguida carregamos novamente:
            model = keras.models.load_model("lstm_model.keras", safe_mode=False)
            
            # Realizamos a previsão em cima dos dados de validação
            val_array = np.array(val_df['Close'], dtype=np.float32)
            val_rnn_forecast = model.predict(val_array[np.newaxis, :, np.newaxis])
            val_rnn_forecast = val_rnn_forecast[0, :len(val_array), 0]
            
            #Plotamos os dados de validação e os dados previstos
            plt.figure(figsize=(10, 6))
            plot_series(time=val_df.index, series=val_df['Close'])
            plot_series(time=val_df.index, series=val_rnn_forecast
            
            #Calculamos a performance do modelo usando a métrica de MeanAbsoluteError
            mae = keras.metrics.MeanAbsoluteError()
            mae.update_state(x_valid, val_rnn_forecast)
            mae.result().numpy()
            
            #Calculamos a performance do modelo usando a métrica do MeanSquareError
            mse = keras.metrics.MeanSquaredError()
            mse.update_state(x_valid, val_rnn_forecast)
            mse.result().numpy()
            
            #Realizamos novamente a previsão usando os dados de teste
            test_array = np.array(test_df['Close'], dtype=np.float32)
            test_rnn_forecast = model.predict(test_array[np.newaxis, :, np.newaxis])
            test_rnn_forecast = test_rnn_forecast[0, :len(test_array), 0]
            
            #Plotamos os dados de teste com a previsão
            plt.figure(figsize=(10, 6))
            plot_series(time=test_df.index, series=test_df['Close'])
            plot_series(time=test_df.index, series=test_rnn_forecast)
            
            #Calculamos a performance do modelo usando a métrica de MeanAbsoluteError
            mae = keras.metrics.MeanAbsoluteError()
            mae.update_state(x_test, test_rnn_forecast)
            mae.result().numpy()
            
            #Calculamos a performance do modelo usando a métrica do MeanSquareError
            mse = keras.metrics.MeanSquaredError()
            mse.update_state(x_test, test_rnn_forecast)
            mse.result().numpy()
        """
        st.code(codigo_python, language='python')

    with tab4:

        st.markdown("""Abaixo temos a previsão realizada pelo modelo do período que vai de 2021 à 2024. Como já temos conhecimento dos valores de fechamento desse período é possível compará-los aos valores previstos pelo modelo.""")

        model = load_lstm_model()

        df = get_new_data()

        df = df.drop(columns=['Log Returns']) # Removendo colunas desnecessárias

        train_df, val_df, test_df = split_data(df)
        # test_df = df[-30:]
        s_train_df, s_val_df, s_test_df = standardize_data(train_df, val_df, test_df)

        test_array = np.array(s_test_df['Close'], dtype=np.float32)
        test_rnn_forecast = model.predict(test_array[np.newaxis, :, np.newaxis])
        test_rnn_forecast = test_rnn_forecast[0, :len(test_array), 0]

        import plotly.graph_objs as go 

        novoDF = pd.DataFrame(test_df)
        novoDF['Close'] = test_rnn_forecast

        train_df, val_df, u_novoDF = unstandardize_data(train_df, s_train_df, s_val_df, novoDF)

        fig = px.line(test_df, x=test_df.index, y='Close', title='Série temporal de fechamento')
        fig.update_xaxes(title_text='Data')
        fig.update_yaxes(title_text='Cotação')
        fig.add_trace(go.Scatter(x=u_novoDF.index, y=u_novoDF['Close'], mode='lines', name='Previsão RNN'))

        fig.update_layout(width=1200, height=700)

        st.plotly_chart(fig)

        st.markdown("""É válido lembrar, como foi citado na aba de insights, que eventos de grande magnitude podem causar um impacto significativo na economia global e consequentemente na variação do preço do Petróleo, dificultando a previsão e diminuindo a acertividade do modelo.""")

if __name__ == "__main__":
    run()
#%%
