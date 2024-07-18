from flask import Flask, render_template, request, session, jsonify
import google.generativeai as genai
import os
from gemini.promts import instruccion2,documents
import google.generativeai as genai
import tempfile
import pandas as pd
import numpy as np
from IPython.display import display
from IPython.display import Markdown
import textwrap
from markupsafe import Markup
import re
import markdown2

generation_config = {
  "temperature": 0.8,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 800,
}

safety_settings = [
     {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]
model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

model_embedding = 'models/text-embedding-004'
global chat
chat = model.start_chat(history=[])
instruction = instruccion2
AUDIO_FOLDER = 'templates/audio_files'
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key='AQUI_VA_TU_API_KEY')

def format_response(text):
    # Convertir Markdown a HTML
    html = markdown2.markdown(text)

    # Eliminar los asteriscos extras que no se convirtieron
    html = re.sub(r'\*\*?', '', html)

    # Convertir los guiones al principio de la línea en viñetas HTML
    html = re.sub(r'^\s*- ', '• ', html, flags=re.MULTILINE)

    return html

def reset_conversation():
    # Limpiar la caché de embeddings
    if os.path.exists('embeddings_cache.pkl'):
        os.remove('embeddings_cache.pkl')

    # Reiniciar el chat
    global chat
    chat = model.start_chat(history=[])

    # Limpiar el historial de conversación en la sesión
    session['conversation_history'] = []

def clean_text(text):
    """
    Función para limpiar el texto y eliminar las etiquetas HTML.
    """
    cleaned_text = re.sub(r'<[^>]+>|\*|#', '', text)
    return cleaned_text

app = Flask(__name__, static_folder='templates')
app.secret_key = 'su_clave_secreta_segura_aquí'

def embed_fn(title, text):
    return genai.embed_content(model=model_embedding,
                               content=text,
                               task_type="retrieval_document",
                               title=title)["embedding"]

def find_best_passages(query, dataframe, top_n=3):
    query_embedding = genai.embed_content(model=model_embedding,
                                content=query,
                                task_type="retrieval_query")
    dataframe['Embeddings'] = dataframe.apply(lambda row: embed_fn(row['Title'], row['Text']), axis=1)
    dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
    top_indices = np.argsort(dot_products)[-top_n:][::-1]
    return dataframe.iloc[top_indices]['Text'].tolist()

MAX_HISTORY = 5

def update_conversation_history(history, user_input, bot_response):
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": bot_response})
    return history[-MAX_HISTORY*2:]

def reset_conversation():
    # Limpiar la caché de embeddings
    if os.path.exists('embeddings_cache.pkl'):
        os.remove('embeddings_cache.pkl')

    # Reiniciar el chat
    global chat
    chat = model.start_chat(history=[])

    # Limpiar el historial de conversación en la sesión
    if 'conversation_history' in session:
        session['conversation_history'] = []

@app.route('/', methods=['GET', 'POST'])
def home():
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    if request.method == 'GET':
        reset_conversation()
        return render_template('index.html', conversations=[])

    elif request.method == 'POST':
        question = request.form['question'].lower()

        # Si no contiene las palabras clave, continuar con el procesamiento normal
        df = pd.DataFrame(documents)
        df.columns = ['Title', 'Text']

        passages = find_best_passages(question, df)
        context = "\n".join(passages)

        full_prompt = f"{instruccion2}\n\nContext:\n{context}\n\nConversation history:\n"
        for entry in session['conversation_history']:
            full_prompt += f"{entry['role']}: {entry['content']}\n"
        full_prompt += f"user: {question}\nassistant:"

        response = chat.send_message(full_prompt)
        bot_response = response.text

        session['conversation_history'] = update_conversation_history(
            session['conversation_history'], question, bot_response)

        print("Input tokens:", len(full_prompt.split()))
        print("Output tokens:", len(bot_response.split()))

        formatted_response = format_response(bot_response)
        response_lines = [Markup(line) for line in formatted_response.split('\n') if line.strip()]

        session.modified = True
        return jsonify({'response': response_lines})

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 4000)))