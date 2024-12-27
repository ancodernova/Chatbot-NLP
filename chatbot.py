import json
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from colorama import Fore, Style
import random
import matplotlib.pyplot as plt
import os

# Load intents
with open('intents.json', encoding='utf-8') as file:
    data = json.load(file)

# Prepare training data
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Sequence and pad sentences
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=20)

# Model parameters
vocab_size = len(word_index) + 1
embedding_dim = 16
max_len = 20
num_classes = len(labels)

# Build model
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

model.summary()

# Train model
epochs = 100
history = model.fit(padded_sequences,
                    np.array(training_labels),
                    epochs=epochs,
                    verbose=2)

# Save model and preprocessing objects
model.save("chat_model.h5")  # Save with .h5 extension

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('label_encoder.pickle', 'wb') as ecn_file:
    pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

# Plot training accuracy and loss
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Chat function with dynamic learning
def chat():
    model = tf.keras.models.load_model('chat_model.h5')  # Load .h5 model
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    print(Fore.YELLOW + "Start chatting with the bot (type 'quit' to exit)!" + Style.RESET_ALL)
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        # Predict response
        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        if tag[0] in responses:
            print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL, random.choice(responses[tag[0]]))
        else:
            print(Fore.RED + "ChatBot: I didn't understand that." + Style.RESET_ALL)

        # Ask user if they want to add the conversation for learning
        print(Fore.YELLOW + "Do you want to add this conversation to improve the chatbot? (yes/no)" + Style.RESET_ALL, end=" ")
        user_input = input()
        if user_input.lower() == "yes":
            # Add new pattern and label to the training data
            new_intent = input(Fore.LIGHTCYAN_EX + "Enter the intent tag for this conversation: " + Style.RESET_ALL)
            data['intents'].append({
                'tag': new_intent,
                'patterns': [inp],
                'responses': ['Thank you for your input!']
            })
            # Retrain the model
            retrain_model(data)

def retrain_model(data):
    # Retrain the model with new data
    training_sentences = []
    training_labels = []
    labels = []
    responses = {}

    for intent in data['intents']:
        for pattern in intent['patterns']:
            training_sentences.append(pattern)
            training_labels.append(intent['tag'])
        responses[intent['tag']] = intent['responses']

        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # Encode labels
    lbl_encoder = LabelEncoder()
    training_labels = lbl_encoder.fit_transform(training_labels)

    # Tokenize sentences
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index

    # Sequence and pad sentences
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded_sequences = pad_sequences(sequences, truncating='post', maxlen=20)

    # Model parameters
    vocab_size = len(word_index) + 1
    embedding_dim = 16
    max_len = 20
    num_classes = len(labels)

    # Build and compile model
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    # Train model
    model.fit(padded_sequences,
              np.array(training_labels),
              epochs=10,
              verbose=2)

    # Save the model and updated tokenizer
    model.save("chat_model.h5")
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('label_encoder.pickle', 'wb') as ecn_file:
        pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)

    print(Fore.GREEN + "Model retrained with new data." + Style.RESET_ALL)

# Start the chat
chat()
