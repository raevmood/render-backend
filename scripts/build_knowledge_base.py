import os
from utils.helper import pdf_loader, create_chunk,download_embbeding_model
from langchain_community.vectorstores import Chroma


def main():
    metadata =[]
    
    documents = pdf_loader("data/raw")
    text_chunk = create_chunk(documents)
    embeddings = download_embbeding_model()
    persist_dir="data/chroma"

    docsearch = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    # Save to local
    docsearch.persist()
    print("✅ Knowledge Base built and stored in ChromaDB.")



if __name__ == "__main__":
    main()
