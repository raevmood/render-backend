import os
from utils.helper import txt_loader, create_chunk,download_embbeding_model
from langchain_community.vectorstores import Chroma


def main():
    metadata =[]
    
    documents = txt_loader("data/raw")
    text_chunk = create_chunk(documents)
    embeddings = download_embbeding_model()
    persist_dir="data/chroma"

    docsearch = Chroma.from_documents(
        documents=text_chunk,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    # Save to local
    docsearch.persist()
    print("✅ Knowledge Base built and stored in ChromaDB.")



if __name__ == "__main__":
    main()
