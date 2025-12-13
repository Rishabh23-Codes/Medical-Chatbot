from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load Raw PDF
DATA_PATH="Data/"

DB_FAISS_PATH='vectorstore/db_faiss'

def load_pdf_files(data):
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)

    return loader.load()


# Create chunks 
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    return text_splitter.split_documents(extracted_data)


# Create VectorStore and save it to locally so that it can use in main.py file 

def create_vectorstore(chunks):
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.from_documents(chunks,embedding_model)
    db.save_local(DB_FAISS_PATH)
    print("✅ FAISS vector store saved at:", DB_FAISS_PATH)


if __name__=='__main__':
    documents=load_pdf_files(data=DATA_PATH)
    text_chunks=create_chunks(extracted_data=documents)
    create_vectorstore(text_chunks)

