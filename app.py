# Para crear el requirements.txt ejecutamos 
# pipreqs --encoding=utf8 --force

# Primera Carga a Github
# git init
# git add .
# git commit -m "primer commit"
# git remote add origin https://github.com/nicoig/chat-jacobo.git
# git push -u origin master

# Actualizar Repo de Github
# git add .
# git commit -m "Se actualizan las variables de entorno"
# git push origin master

# Para eliminar un repo cargado
# git remote remove origin

# En Render
# agregar en variables de entorno
# PYTHON_VERSION = 3.9.12

# Pasando a master
# git checkout -b master
# git push origin master



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
        Eres el psicólogo y antropólogo Jacobo Grinberg. Se te proporcionarán varios documentos de texto basados
        en las investigaciones de Grinberg que eres tu, debes tomar el rol y se espera que respondas preguntas relacionadas con ellos de la manera 
        más clara y concisa posible. Si no tienes la respuesta, simplemente di que no la sabes en lugar de intentar adivinarla. 
        Si la pregunta no está relacionada con las investigaciones de Grinberg, cortésmente señala que estás aquí para responder 
        preguntas relacionadas con su trabajo. Utiliza los fragmentos de contexto a continuación para formular tu respuesta.

        context: {context}
        =========
        question: {question}
        ======
        """
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context","question" ])
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={'prompt': QA_PROMPT}
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            

def main():
    load_dotenv()
    st.set_page_config(page_title="Chatea con Jacobo Grinberg", page_icon=":books:", layout="wide")
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
    st.header("Chat con Jacobo Grinberg :books:")

    # Estableciendo el subtítulo
    #st.subheader("Chatea, explora y aprende de forma dinámica")

        # Mostrar la imagen
    st.image('img/jacobo_3.jpg', width=500)


    st.write("""
    Soy Jacobo Grinberg, conocido por mi trabajo en psicología y antropología con un enfoque particular en la conciencia y la percepción. A través de este chatbot, puedes consultar sobre mis trabajos, proyectos e ideas, tales como:

    - La integración de datos fisiológicos en un cuerpo teórico comprensivo y racional.
    - Mi teoría sobre que todo lo que existe es un nivel particular de conciencia, incluso la materia.
    - Mis análisis detallados de la naturaleza del "yo" desde un punto de vista racional y lógico.
    - Mis exploraciones en torno a temas complejos y profundos relacionados con el amor, la libertad y la psicofisiología.
""")


    user_question = st.text_input("Realiza tu consulta:")
    if st.button('Enviar'):  
        if user_question:
            handle_userinput(user_question)


if __name__ == '__main__':
    main()
