import os
import sys
import pandas as pd
import string
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
import tensorflow_hub
import random
import numpy as np
import config
import pickle

base_dir = os.path.dirname(__file__)
tokenizer_file_path = os.path.join(base_dir, "tokenizer.pickle")
pad = tf.keras.preprocessing.sequence.pad_sequences


def remover_pontos(texto):
    texto = str(texto)
    return texto.translate(
        str.maketrans("", "", string.punctuation)
    )


def remover_palavras_de_parada(texto):
    texto = str(texto)
    parada = set(stopwords.words("english"))
    novo_texto_lista = []
    for palavra in texto.split():
        if palavra.lower() not in parada:
            novo_texto_lista.append(palavra.lower())
    novo_texto = " ".join(novo_texto_lista)
    return novo_texto


def get_x_and_y(df, tokenizer):
    texts = []
    x, y = [], []

    for i in range(len(df.iloc[:])):
        texts.append(df["v2"].iloc[i])
        y1 = int(df["v1"].iloc[i])
        y.append(y1)
    x = pad(tokenizer.texts_to_sequences(texts),
            maxlen=config.pad_length, padding="post", truncating="post")
    return x, y


def shuffle_data(dataset):
    x, y = dataset
    data = []
    for i in range(len(x)):
        data.append({
            "input": x[i],
            "output": y[i]
        })
    random.shuffle(data)
    x = [data[i]["input"] for i in range(len(data))]
    y = [data[i]["output"] for i in range(len(data))]
    return x, y


def clearcsv():
    csvfile = pd.read_csv(os.path.join(
        base_dir, "dataset/spam.csv"), encoding="ISO-8859-1")
    ham_df = csvfile[csvfile["v1"] == "ham"]
    spam_df = csvfile[csvfile["v1"] == "spam"].iloc[:]
    ham_df = ham_df.iloc[:len(spam_df.iloc[:])]
    for i in range(len(ham_df.iloc[:])):
        ham_df["v1"] = ham_df["v1"].replace([ham_df["v1"].iloc[i]], 0)
    for i in range(len(spam_df.iloc[:])):
        spam_df["v1"] = spam_df["v1"].replace([spam_df["v1"].iloc[i]], 1)
    split_size = len(ham_df.iloc[:]) + len(spam_df.iloc[:])
    split_size = int(split_size * 0.8)
    split_size = int(split_size/2)
    train_cv = pd.concat([ham_df.iloc[:split_size], spam_df.iloc[:split_size]])
    test_cv = pd.concat([ham_df.iloc[split_size:], spam_df.iloc[split_size:]])
    train_cv.to_csv(os.path.join(base_dir, "dataset/train_df.csv"))
    test_cv.to_csv(os.path.join(base_dir, "dataset/test_df.csv"))
    return


def clean_df(df):
    for i in range(len(df.iloc[:])):
        texto = df["v2"].iloc[i]
        texto = remover_pontos(texto)
        texto = remover_palavras_de_parada(texto)
        df["v2"] = df["v2"].replace([df["v2"].iloc[i]], texto)
    return df


def tokenize_data(train_df):
    text_sequence = []
    for i in range(len(train_df.iloc[:])):
        text = train_df["v2"].iloc[i]
        text_sequence.append(text)
    tokenizer = Tokenizer(config.vocab_size)
    tokenizer.fit_on_texts(text_sequence)
    with open(tokenizer_file_path, "wb") as file:
        pickle.dump(tokenizer, file)
    return tokenizer


def normalizar_texto(texto):
    tokenizer = ""
    if os.path.exists(tokenizer_file_path):
        with open(tokenizer_file_path, "rb") as file:
            tokenizer = pickle.load(file)
    texto = remover_pontos(texto)
    texto = remover_palavras_de_parada(texto)
    n_texto = tokenizer.texts_to_sequences([texto])
    n_texto = pad(n_texto, maxlen=config.pad_length,
                  padding="post", truncating="post")
    return n_texto


def get_dataset():
    train_df = pd.read_csv(os.path.join(base_dir, "dataset/train_df.csv"))
    test_df = pd.read_csv(os.path.join(base_dir, "dataset/test_df.csv"))
    train_df = clean_df(train_df)
    test_df = clean_df(test_df)
    main_df = pd.concat([train_df, test_df])
    tokenizer = tokenize_data(main_df)
    train_dataset = get_x_and_y(train_df, tokenizer)
    train_dataset = shuffle_data(train_dataset)
    test_dataset = get_x_and_y(test_df, tokenizer)
    return train_dataset, test_dataset
