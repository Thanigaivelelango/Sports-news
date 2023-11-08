from flask import Flask, render_template, request
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the tokenizer and model
tokenizer = Tokenizer()
tokenizer.fit_on_texts("your test samples")  # Assuming 'df' is your original DataFrame
max_sequence_length = 100  # Assuming the same sequence length used during training
model = load_model('model.h5')  # Load your trained model file
categories=['Cricket', 'Football', 'Basketball', 'Tennis', 'American Football',
       'Baseball', 'Rugby', 'Golf', 'Ice Hockey', 'Boxing', 'MMA',
       'Formula1', 'Cycling', 'Wrestling', 'Athletics', 'Swimming',
       'Gymnastics', 'Volleyball', 'Table Tennis', 'Badminton',
       'Tae Kwon Do', 'Skiing', 'Snowboarding', 'Surfing', 'Esports',
       'Horse Racing', 'Fencing', 'Archery']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        headline = request.form['headline']
        # Tokenize and pad the input text
        input_text_sequences = tokenizer.texts_to_sequences([headline])
        input_text_padded = pad_sequences(input_text_sequences, maxlen=max_sequence_length, padding='post')
        # Make a prediction using the loaded model
        predicted_probabilities = model.predict(input_text_padded)
        # Decode the predicted category
        predicted_category_index = predicted_probabilities.argmax(axis=1)[0]
        predicted_category = categories[predicted_category_index]  # Define 'categories' based on your dataset
        return render_template('index.html', headline=headline, prediction=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)
