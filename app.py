import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub  # بديل مجاني لـ OpenAI

# واجهة المستخدم
st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("🤖 اسأل ملف PDF")

# تحميل الملف
pdf = st.file_uploader("ارفع ملف PDF", type="pdf")

# معالجة الملف
if pdf:
    with open("temp.pdf", "wb") as f:
        f.write(pdf.read())

    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)

    query = st.text_input("💬 اكتب سؤالك:")
    if query:
        docs = db.similarity_search(query)
        chain = load_qa_chain(llm=HuggingFaceHub(repo_id="google/flan-t5-base"), chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)
        st.write("### ✅ الإجابة:")
        st.write(answer)
