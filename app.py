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
        page_title="Hello",
        page_icon="👋",
        layout="wide"
    )
    st.title(":chart_with_upwards_trend: Previsão do preço do Petróleo Brent")

    st.write("# Tech Challenge Fase 4 ")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
## Prevendo os valores futuros do petróleo Brent

Membros do grupo 24:

- RM351388 - Carolina Pasianot Casetta - carol_pasianot@hotmail.com
- RM351418 - Gustavo França Severino - gustavofs.dt@gmail.com
- RM352372 - Frederico Garcia Costa - fredgcosta@gmail.com
- RM351905 - Jeferson Vieira - jvieirax@gmail.com
- Victor Wilson Costa Lamana - victor.lamana15@gmail.com

Neste projeto iremos realizar uma análise da série temporal do petróleo Brent e propor uma estratégia para a previsão dos valores futuros. Iremos utilizar as bibliotecas TensorFlow e Keras para a criação dos modelos.
    """
    )

    model = load_lstm_model()

    df = get_data()

    df = df.drop(columns=['Log Returns']) # Removendo colunas desnecessárias

    train_df, val_df, test_df = split_data(df)
    train_df, val_df, test_df = standardize_data(train_df, val_df, test_df)

    val_array = np.array(val_df['Close'], dtype=np.float32)
    val_rnn_forecast = model.predict(val_array[np.newaxis, :, np.newaxis])
    val_rnn_forecast = val_rnn_forecast[0, :len(val_array), 0]

    fig = plt.figure(figsize=(10, 6))
    plot_series(time=val_df.index, series=val_df['Close'])
    plot_series(time=val_df.index, series=val_rnn_forecast)

    st.pyplot(fig)

if __name__ == "__main__":
    run()
#%%
