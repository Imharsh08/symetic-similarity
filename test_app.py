import pytest
from flask import Flask, render_template_string
from your_app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b"Semantic Similarity Score App" in response.data

def test_calculate_similarity(client):
    data = {'sentence1': 'Hello', 'sentence2': 'World'}
    response = client.post('/calculate_similarity', data=data)
    assert response.status_code == 200
    assert b"Result" in response.data
    assert b"Similarity Score:" in response.data
    assert b"Text 1:" in response.data
    assert b"Text 2:" in response.data

if __name__ == '__main__':
    pytest.main()
