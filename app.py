from flask import Flask, request, jsonify, render_template
from transformers import AutoModel
from numpy.linalg import norm

app = Flask(__name__)


# Define cosine similarity function
def cos_similarity(a, b):
    return (a @ b.T) / (norm(a) * norm(b))


# Load Jina Embeddings model from Hugging Face Transformers
model = AutoModel.from_pretrained("jinaai/jina-embeddings-v2-base-en", trust_remote_code=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    data = request.get_json()
    sentences = data.get('sentences', [])

    # Encode sentences and compute embeddings
    embeddings = model.encode(sentences)

    # Calculate cosine similarity between the embeddings
    similarity = cos_similarity(embeddings[0], embeddings[1])

    response = {
        'cosine_similarity': similarity.tolist()
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
