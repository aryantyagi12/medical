import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def build_rag_chain():
    # 1. Embeddings (runs locally — small & fast)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2. Connect to Pinecone
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # 3. LLM via Groq (free, fast — uses GROQ_API_KEY env var)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=300,
    )

    # 4. Prompt
    prompt = PromptTemplate(
        template="""Answer the question using ONLY the context below.

Rules:
- Give a clear, direct answer.
- Do not repeat information.
- Answer in 2-4 sentences maximum.
- If the answer is not in the context, say: "I couldn't find that in the document."

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"]
    )

    # 5. Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Input must be a dict: {"question": "<user question>"}
    # itemgetter pulls the question string out for the retriever and prompt
    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain_tuple, question: str) -> dict:
    chain, retriever = chain_tuple
    # Pass a dict so itemgetter("question") works correctly inside the chain
    answer = chain.invoke({"question": question})
    source_docs = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": [doc.metadata for doc in source_docs]
    }