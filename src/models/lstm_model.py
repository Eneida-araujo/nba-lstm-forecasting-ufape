"""
=========================================================
lstm_model.py
=========================================================

Responsável por construir, compilar e treinar modelos LSTM.

A arquitetura foi criada para previsão numérica de:
- pontos (PTS);
- rebotes (REB);
- assistências (AST).

Inclui:
- camadas LSTM;
- Dense;
- Dropout;
- recurrent_dropout;
- kernel_regularizer;
- função de perda;
- otimizador Adam;
- gradient clipping;
- callbacks contra overfitting.
=========================================================
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


class LSTMModelBuilder:
    """
    Classe responsável pela criação e treinamento da LSTM.
    """

    def __init__(
        self,
        input_shape: tuple,
        learning_rate: float = 0.001,
        clipnorm: float = 1.0,
        l2_value: float = 0.001
    ):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.clipnorm = clipnorm
        self.l2_value = l2_value
        self.model = None

    def build_model(self):
        """
        Constrói a arquitetura da LSTM.

        input_shape:
        (timesteps, features)

        Exemplo:
        (5, 38)
        significa:
        - 5 jogos anteriores;
        - 38 features por jogo.
        """

        model = Sequential()

        model.add(
            LSTM(
                units=64,
                return_sequences=True,
                recurrent_dropout=0.10,
                kernel_regularizer=l2(self.l2_value),
                input_shape=self.input_shape
            )
        )

        model.add(
            Dropout(0.20)
        )

        model.add(
            LSTM(
                units=32,
                return_sequences=False,
                recurrent_dropout=0.10,
                kernel_regularizer=l2(self.l2_value)
            )
        )

        model.add(
            Dropout(0.20)
        )

        model.add(
            Dense(
                units=16,
                activation="relu",
                kernel_regularizer=l2(self.l2_value)
            )
        )

        model.add(
            Dense(
                units=1,
                activation="linear"
            )
        )

        optimizer = Adam(
            learning_rate=self.learning_rate,
            clipnorm=self.clipnorm
        )

        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )

        self.model = model

        return model

    def get_callbacks(self):
        """
        Define callbacks para reduzir overfitting.

        EarlyStopping:
        interrompe o treinamento quando o modelo para de melhorar.

        ReduceLROnPlateau:
        reduz a taxa de aprendizado quando a validação estabiliza.
        """

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True
        )

        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=0.00001
        )

        return [early_stopping, reduce_lr]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Treina o modelo LSTM.
        """

        if self.model is None:
            self.build_model()

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.get_callbacks(),
            verbose=verbose
        )

        return history

    def predict(self, X: np.ndarray):
        """
        Gera previsões.
        """

        if self.model is None:
            raise ValueError(
                "O modelo ainda não foi construído ou treinado."
            )

        predictions = self.model.predict(X)

        return predictions.flatten()

    def summary(self):
        """
        Exibe resumo da arquitetura.
        """

        if self.model is None:
            self.build_model()

        self.model.summary()