# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git commit -m "primer commit"
# git remote add origin https://github.com/nicoig/legalmind.git
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# Para eliminar un repo cargado
# git remote remove origin

# Pasando a master
# git checkout -b master
# git push origin 

# Verificando donde está montado el proyecto
# git remote -v

# Cambiar la URL del control remoto "origin":
# git remote set-url origin https://github.com/nicoig/legalmind.git





###############################################################



import os
import pickle
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.prompts.prompt import PromptTemplate


import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(filepaths):
    text = ""
    for filepath in filepaths:
        with open(filepath, "rb") as file:
            pdf = PdfReader(file)
            for page in pdf.pages:
                text += page.extract_text()
    return text

def get_text_chunks(text, chunks_file):
    if os.path.exists(chunks_file):
        with open(chunks_file, 'rb') as f:
            chunks = pickle.load(f)
    else:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        with open(chunks_file, 'wb') as f:
            pickle.dump(chunks, f)
    return chunks

def get_vectorstore(text_chunks, vectorstore_file):
    if os.path.exists(vectorstore_file):
        with open(vectorstore_file, 'rb') as f:
            vectorstore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        with open(vectorstore_file, 'wb') as f:
            pickle.dump(vectorstore, f)
    return vectorstore

def get_conversation_chain(vectorstore, model_name):
    llm = ChatOpenAI(model_name=model_name)
    qa_template = """
        Eres un abogado experto en asesoramiento legal llamado LegalMind. Se te proporcionarán varios documentos de texto basados
        en contextos legales específicos. Debes tomar el rol de un asesor legal y se espera que respondas preguntas relacionadas 
        con estos contextos de la manera más clara y concisa posible. Si no tienes la respuesta, simplemente di que no la 
        sabes en lugar de intentar adivinarla. Si la pregunta no está relacionada con el contexto legal proporcionado, 
        cortésmente señala que estás aquí para responder preguntas relacionadas con ese ámbito legal. Utiliza los fragmentos de 
        contexto a continuación para formular tu respuesta.

        Contexto: {context}
        =========
        Pregunta: {question}
        ======
    """
    # Aquí podrías añadir el código para generar la respuesta usando el modelo.

    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT}
    )
    return conversation_chain


def handle_userinput(user_question, chat_placeholder):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    chat_content = ""
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            chat_content += user_template.replace("{{MSG}}", message.content)
        else:
            chat_content += bot_template.replace("{{MSG}}", message.content)
    
    chat_placeholder.write(chat_content, unsafe_allow_html=True)  # Actualizar el chat en el espacio vacío


            

def main():
    load_dotenv()
    st.set_page_config(page_title="LegalMind - Abogado IA", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)

    st.sidebar.title('Menu')
    model_name = st.sidebar.selectbox(
        'Selecciona un modelo de LLM:',
        ('gpt-3.5-turbo', 'gpt-3.5-turbo-16k','text-davinci-003','gpt-4') # Puedes poner los modelos que quieras aquí
    )
    temperature = st.sidebar.slider('Ajusta la temperatura:', min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    if "initialized" not in st.session_state:
        file_directory = 'files'
        filepaths = [os.path.join(file_directory, file) for file in os.listdir(file_directory) if file.endswith('.pdf')]
        text = get_pdf_text(filepaths)

        chunks_file = 'chunks.pkl'
        chunks = get_text_chunks(text, chunks_file)

        vectorstore_file = 'vectorstore.pkl'
        vectorstore = get_vectorstore(chunks, vectorstore_file)

        st.session_state.conversation = get_conversation_chain(vectorstore, model_name)
        st.session_state.llm_temperature = temperature
        st.session_state.chat_history = []
        st.session_state.initialized = True


    # Estableciendo el título
    st.header("🤖⚖️ LegalMind - Abogado IA ⚖️🤖")

    # Estableciendo el subtítulo
    #st.subheader("Chatea, explora y aprende de forma dinámica")

        # Mostrar la imagen
    st.image('img/abogado.jpg', width=500)


    st.write("""
    Soy LegalMind, tu Asistente Legal Inteligente. Estoy programado para ofrecer información y asistencia en una variedad de contextos legales, tales como:

    - Interpretación básica de leyes y estatutos.
    - Información general sobre procesos legales, como juicios y apelaciones.
    - Consejos preliminares sobre cómo abordar situaciones legales específicas.
    - Respuestas a preguntas frecuentes en el ámbito del derecho.
""")

    chat_placeholder = st.empty()  # Crea un espacio vacío para el chat

    user_question = st.chat_input("Realiza tu consulta:")
    if user_question:
        handle_userinput(user_question, chat_placeholder)  # Pasar chat_placeholder como argumento



if __name__ == '__main__':
    main()
