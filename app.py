from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Load pre-trained BERT model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    # Ensure to use the correct form field names from your HTML form
    sentence1 = request.form.get('sentence1', '')
    sentence2 = request.form.get('sentence2', '')

    # Encode the sentences
    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)

    # Calculate cosine similarity between the embeddings
    similarity_score = util.pytorch_cos_sim(embeddings1, embeddings2).item()

    # Prepare the result
    result = {
        'similarity_score': similarity_score,
        'sentence1': sentence1,
        'sentence2': sentence2
    }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)