from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def txt_loader(docs: str):
    """
    Load all .txt files from the specified directory.
    """
    loader = DirectoryLoader(
        docs,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"} 
    )
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