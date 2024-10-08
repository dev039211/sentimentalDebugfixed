import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    return app.test_client()

def load_keras_model():
    global model
    model = load_model('models/uci_sentimentanalysis.h5')

def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

def test_load_keras_model():
    load_keras_model()
    assert model is not None

def test_load_tokenizer():
    load_tokenizer()
    assert tokenizer is not None

def test_sentiment_analysis():
    input = "I love python"
    expected_output = 0.9
    load_keras_model()
    load_tokenizer()
    user_sequences = tokenizer.texts_to_sequences([input])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    assert prediction[0][0] >= expected_output

def test_index_get(client):
    response = client.get('/')
    assert response.status_code == 200

def test_index_post_positive_sentiment(client):
    response = client.post('/', data={'user_text': 'I love python'})
    assert response.status_code == 200
    assert b"Positive:" in response.data

def test_index_post_negative_sentiment(client):
    response = client.post('/', data={'user_text': 'I hate python'})
    assert response.status_code == 200
    assert b"Negative:" in response.data