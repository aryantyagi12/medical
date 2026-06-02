import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

load_dotenv()

def build_rag_chain():
    # 1. Embeddings
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

    # 3. Local LLM — facebook/opt-125m runs on CPU, no API needed
    model_id = "facebook/opt-125m"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 4. Prompt
    prompt = PromptTemplate(
        template = """
Answer the question using the context below.

Rules:
- Give only one answer.
- Do not repeat information.
- Answer in 2-3 sentences maximum.
- If the answer is not present, say:
  "I couldn't find that in the document."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    # 5. Chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain_tuple, question: str) -> dict:
    chain, retriever = chain_tuple
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": [doc.metadata for doc in source_docs]
    }