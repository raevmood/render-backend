from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def txt_loader(docs: str):
    """
    Load all .txt and .pdf files from the specified directory.
    """
    # Load .txt files
    txt_loader = DirectoryLoader(
        docs,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    txt_documents = txt_loader.load()

    # Load .pdf files
    pdf_loader = DirectoryLoader(
        docs,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    pdf_documents = pdf_loader.load()

    # Combine both
    documents = txt_documents + pdf_documents
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