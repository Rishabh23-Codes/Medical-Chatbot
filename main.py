
import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()  # Load GROQ_API_KEY from .env

# ------------------ Config ------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

# ------------------ Load Vectorstore ------------------
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2'
    )
    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

# ------------------ Streamlit Frontend ------------------
def main():
    st.title('Ask MedBot!')

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # User input
    prompt = st.chat_input('Enter your question here:')

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            # Load FAISS vectorstore
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error('Failed to load the vector store')
                return

            # Initialize GROQ LLM
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
            GROQ_MODEL_NAME = "llama-3.1-8b-instant"

            llm = ChatGroq(
                model=GROQ_MODEL_NAME,
                temperature=0.5,
                max_tokens=512,
                api_key=GROQ_API_KEY
            )

            # Custom RAG PromptTemplate
            retrieval_qa_chat_prompt = PromptTemplate(
                input_variables=["context", "input"],  # Note: 'input' maps to user's prompt
                template="""
You are a helpful assistant. Use ONLY the following context to answer the question. 
If the answer is not contained in the context, respond with "I don't know."

Context:
{context}

Question:
{input}

Answer:
"""
            )

            # Create document combiner chain
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

            # Create retrieval + RAG chain
            rag_chain = create_retrieval_chain(
                vectorstore.as_retriever(search_kwargs={'k': 3}),
                combine_docs_chain
            )

            # Invoke chain with user input
            response = rag_chain.invoke({'input': prompt})
            result = response['answer']

            # Display assistant response
            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f'Error: {str(e)}')

if __name__ == '__main__':
    main()

