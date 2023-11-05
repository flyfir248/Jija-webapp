# Cosine Similarity Calculation using Jina Embeddings model from Hugging Face

This is a simple web application built with Flask and Hugging Face Transformers. It calculates the cosine similarity between two input sentences using a pre-trained Jina Embeddings model from Hugging Face. The application also provides a user-friendly interface for users to input sentences and view the cosine similarity score.

## How it Works

1. **Input Sentences:** Users can enter two sentences into the input fields provided on the web interface.

2. **Calculation:** Upon submitting the form, the application encodes the input sentences using the Jina Embeddings model and computes the cosine similarity between them.

3. **Result:** The computed cosine similarity score is displayed on the web page, providing users with the similarity measure between their input sentences.

## Getting Started

### Prerequisites

Make sure you have Python installed. If not, you can download it from [python.org](https://www.python.org/downloads/).

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/your-username/cosine-similarity-calculator.git
   ```

2. Navigate to the project directory:

   ```
   cd cosine-similarity-calculator
   ```

3. Install the required packages using `pip`:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```
   python app.py
   ```

2. Open your web browser and go to `http://127.0.0.1:5000` to access the Cosine Similarity Calculator.

3. Enter two sentences in the input fields and click the "Calculate Cosine Similarity" button.

4. The application will display the cosine similarity score between the input sentences.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

**Note:** This readme template is just a starting point. Customize it further to fit the specific details and features of your project.