import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢")
st.title("Zyro Dynamics HR Help Desk 🚀")
st.markdown("Ask me anything about our company policies (Leave, WFH, POSH, etc.)")

@st.cache_resource
def load_rag_pipeline():
    # Load the locally saved FAISS index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    
    # Initialize the supported Groq model
    llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"], model_name="llama-3.1-8b-instant", temperature=0)
    
    # Strict Guardrail Prompt
    template = """
    You are the official HR Help Desk Assistant for Zyro Dynamics.
    Use ONLY the provided Context to answer the Question.

    GUARDRAIL: If the question is NOT related to Zyro Dynamics HR policies, or if the answer cannot be found in the context, you MUST refuse by saying EXACTLY: 
    "I can only answer HR-related questions from Zyro Dynamics policy documents."

    Context: {context}

    Question: {question}

    Answer:
    """
    prompt = PromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

rag_chain = load_rag_pipeline()

# Chat UI Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask your HR question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching policies..."):
            response = rag_chain.invoke(prompt)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})