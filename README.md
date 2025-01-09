# Vérifier et installer les bibliothèques nécessaires
try:
    import openai
    import langchain
    import chromadb
    from PyPDF2 import PdfReader
except ImportError:
    !pip install openai langchain chromadb PyPDF2

# Importer les bibliothèques
import os
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Configurer l'API OpenAI (remplacez par votre clé API)
os.environ["OPENAI_API_KEY"] = "VOTRE_CLE_API"  # Remplacez "VOTRE_CLE_API" par votre clé API réelle

# Téléverser un fichier PDF dans Google Colab
from google.colab import files
uploaded = files.upload()  # Chargez un fichier PDF via l'interface

# Charger le document PDF
def charger_documents():
    # Utilisez le nom du fichier téléchargé
    pdf_file = list(uploaded.keys())[0]  # Récupérer le nom du fichier téléversé
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    
    # Créer des embeddings pour la recherche
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore

# Initialiser le stockage vectoriel
vectorstore = charger_documents()

# Configurer le système de questions-réponses
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(model_name="text-davinci-003"),  # Utilisez "gpt-4" si vous avez accès
    retriever=vectorstore.as_retriever()
)

# Fonction de conversation
def discuter():
    print("Bienvenue dans l'IA PDF ! Tapez 'exit' pour quitter.")
    while True:
        user_input = input("Vous : ")
        if user_input.lower() == 'exit':
            print("Au revoir !")
            break

        try:
            # Obtenir une réponse basée sur le PDF
            response = qa_chain.run(user_input)
            print("Bot :", response)
        except Exception as e:
            print("Erreur :", e)

# Lancer la conversation
discuter()
