# ChatBot using TensorFlow and NLP

This project implements an AI-powered chatbot using **TensorFlow**, **Natural Language Processing (NLP)**, and **Deep Learning**. The chatbot is trained to understand and respond to various user inputs based on pre-defined intents provided in a JSON file.

---

## Features

- **Intent Recognition**: The chatbot identifies user intent from input sentences using a trained deep learning model.
- **Custom Responses**: Dynamically responds to user queries based on intent categories defined in the JSON file.
- **Tokenization & Padding**: Preprocesses input sentences using tokenization and sequence padding for efficient model training.
- **Deep Learning Model**: Utilizes TensorFlow's Sequential API with embedding layers and dense layers for intent classification.
- **Interactive Chat**: A command-line chat interface for real-time interactions with the bot.
- **Training Visualization**: Plots model accuracy and loss over training epochs.

---

## Tech Stack

- **Python**: Programming language used for implementation.
- **TensorFlow**: For building and training the deep learning model.
- **NLP Libraries**:
  - `Tokenizer` and `pad_sequences` for text preprocessing.
  - `LabelEncoder` for encoding intent labels.
- **Pickle**: To save and load the tokenizer and label encoder.
- **Matplotlib**: For visualizing training performance.
- **Colorama**: Adds color to console output for better interactivity.

---

## File Descriptions

- **intents.json**: Contains the training data with defined intents, patterns, and responses.
- **chat_model.h5**: Saved model file containing the trained deep learning model.
- **tokenizer.pickle**: Saved tokenizer object for preprocessing new inputs.
- **label_encoder.pickle**: Saved label encoder for decoding predicted intent labels.
- **chatbot.py**: Main Python script for training the model and running the chatbot.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/chatbot.git
   cd chatbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the chatbot:
   ```bash
   python chatbot.py
   ```

---

## Usage

1. Prepare your `intents.json` file with the desired intents, patterns, and responses.
2. Train the chatbot by running the script. The trained model, tokenizer, and label encoder are saved for future use.
3. Interact with the bot in the command-line interface. Type `quit` to exit the chat.

---

## Example Intents JSON

```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi", "Hello", "How are you?"],
      "responses": ["Hello!", "Hi there!", "I'm here to help!"]
    },
    {
      "tag": "goodbye",
      "patterns": ["Bye", "See you", "Goodbye"],
      "responses": ["Goodbye!", "See you later!", "Take care!"]
    }
  ]
}
```

---

## Training Visualization

The model's accuracy and loss during training are plotted and displayed for monitoring performance.

