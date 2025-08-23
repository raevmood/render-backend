from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma



file_path = "docs"
def pdf_loader(docs:str):
    loader = DirectoryLoader(docs,glob="*.pdf",loader_cls=PyPDFLoader )
    documents = loader.load()

    return documents


def create_chunk(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks= text_splitter.split_documents(docs)

    return chunks

def download_embbeding_model():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    return embeddings


def load_chroma_db(embeddings, persist_directory:str):

    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    return vectordb