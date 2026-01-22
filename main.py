import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 1. Page Configuration
st.set_page_config(page_title="Indian AI Financial Auditor", page_icon="🛡️")
st.title("🛡️ Indian Financial & Insurance Auditor")
st.markdown("""
Analyze loan agreements, insurance policies, or bank statements for 
**Red Flags**, **Hidden Fees**, and **Compliance** with RBI/IRDAI rules.
""")

# 2. Handle API Key (From Streamlit Secrets)
# In CodeSandbox, you can also set this as an Environment Variable
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")

if not groq_api_key:
    st.error("Missing Groq API Key! Please add it to Streamlit Secrets or Environment Variables.")
    st.stop()

# 3. File Upload UI
uploaded_file = st.file_uploader("Upload a Financial PDF Document", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with open("temp_audit_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("🔍 Forensic audit in progress..."):
        try:
            # A. Load and Chunking
            loader = PyPDFLoader("temp_audit_doc.pdf")
            documents = loader.load()
            
            # Recursive splitting preserves paragraph context better than simple splitting
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)

            # B. Embeddings (Free & Local)
            # This model runs on the server CPU for free
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # C. Vector Storage (ChromaDB)
            vectorstore = Chroma.from_documents(chunks, embeddings)

            # D. AI Logic Setup (Llama 3 via Groq)
            llm = ChatGroq(
                groq_api_key=groq_api_key, 
                model_name="llama3-8b-8192",
                temperature=0  # 0 for factual, non-creative auditing
            )
            
            # Define the 'Auditor' Persona
            audit_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
            )

            st.success("✅ Document Processed Successfully!")

            # 4. Interactive Auditing
            st.subheader("Ask the Auditor")
            user_q = st.text_input(
                "Example: 'What are the hidden fees?' or 'Is there a free-look period?'",
                placeholder="Type your question here..."
            )

            if user_q:
                # Custom instruction prepended to user query for better 'Audit' focus
                audit_prompt = f"As a financial compliance expert, answer this based on the document: {user_q}"
                response = audit_chain.run(audit_prompt)
                
                st.info("### 🚩 Auditor's Findings:")
                st.write(response)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

# Footer
st.markdown("---")
st.caption("Developed by K chakradhar | Built for the Gytworkz AI Backend Role")
