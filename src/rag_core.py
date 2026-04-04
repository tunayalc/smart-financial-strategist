import os
import torch
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  
from langchain_community.vectorstores import Chroma

KNOWLEDGE_BASE_DIR = "src/knowledge_base"
VECTOR_DB_DIR = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def build_vector_database(force_rebuild=False):
    if os.path.exists(VECTOR_DB_DIR) and not force_rebuild:
        print(f"Mevcut veritabanı bulundu: '{VECTOR_DB_DIR}'")
        return Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=get_embeddings())

    print(f"Belgeler yükleniyor: '{KNOWLEDGE_BASE_DIR}'")
    loader = DirectoryLoader(KNOWLEDGE_BASE_DIR, glob="**/*.txt")
    documents = loader.load()
    if not documents:
        print("Bilgi tabanında belge yok.")
        return None

    print(f"{len(documents)} belge bulundu, parçalanıyor...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"{len(chunks)} parça oluşturuldu.")

    print("Embeddings oluşturuluyor...")
    embeddings = get_embeddings()
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"Veritabanı oluşturuldu: '{VECTOR_DB_DIR}'")
    return vector_db

def get_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device.upper()}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )

def query_rag(db, query, k=2):
    print(f"Sorgu: '{query}'")
    results = db.similarity_search(query, k=k)
    if not results:
        print("Sonuç bulunamadı.")
        return []
    print(f"{len(results)} sonuç bulundu.")
    return results

if __name__ == "__main__":
    vector_database = build_vector_database(force_rebuild=True)

    if vector_database:
        results1 = query_rag(vector_database, "Çeşitlendirmenin faydaları nelerdir?")
        for i, doc in enumerate(results1):
            print(f"\nSonuç {i+1}:\n{doc.page_content}")
            print(f"Kaynak: {doc.metadata.get('source')}")

        results2 = query_rag(vector_database, "Altın neden güvenli liman olarak görülür?")
        for i, doc in enumerate(results2):
            print(f"\nSonuç {i+1}:\n{doc.page_content}")
            print(f"Kaynak: {doc.metadata.get('source')}")