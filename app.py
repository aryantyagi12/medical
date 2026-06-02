import streamlit as st
from ingest import ingest_pdf
from rag_chain import build_rag_chain, ask

# ✅ Set your PDF path here
PDF_PATH = r"C:\Users\Aryan\Documents\med\Medical_book.pdf"

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")
st.caption("Powered by LangChain · Pinecone · HuggingFace")

# ✅ Build chain ONCE per session — never delete it
if "chain" not in st.session_state:
    with st.spinner("Indexing PDF and loading model..."):
        ingest_pdf(PDF_PATH)
        st.session_state["chain"] = build_rag_chain()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if question := st.chat_input("Ask something about your PDF..."):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(st.session_state["chain"], question)
            answer = response["answer"]
            st.write(answer)

            with st.expander("📎 Source chunks"):
                for src in response["sources"]:
                    st.write(f"- Page {src.get('page', '?')} | {src.get('source', '')}")

    st.session_state["messages"].append({"role": "assistant", "content": answer})