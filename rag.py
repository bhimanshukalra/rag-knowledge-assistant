from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_community.retrievers import TFIDFRetriever
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

documents = [
    Document(
        page_content="LangChain is a framework for building applications with large language models. It helps with chains, agents, retrieval, and integrations."
    ),
    Document(
        page_content="RAG stands for Retrieval-Augmented Generation. It first retrieves relevant information, then gives that information to an LLM so it can answer with context."
    ),
    Document(
        page_content="Gemini is Google's family of AI models. In LangChain, Gemini can be used through the langchain-google-genai package."
    ),
]

retriever  = TFIDFRetriever.from_documents(documents)
retriever.k = 2

llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

def ask(question):
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)
    prompt = f"""
Answer the question using only the context below.

Context:
{context}
Question:
{question}

Answer:
"""
    response = llm.invoke(prompt)

    return response.content

question = input("Ask a question: ")
answer = ask(question)

print("\nAnswer: ")
print(answer)