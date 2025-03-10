import streamlit as st
import pickle 
import time
import os
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 
import pypdf
from pypdf import PdfReader
from langchain_groq import ChatGroq
from langchain.schema import Document

os.environ["GROQ_API_KEY"] = "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"
# Initialize the ChatGroq model
llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0.6,max_tokens=500)

# Set up Streamlit page configuration
st.set_page_config(page_title="PDF Question Answer Tool", page_icon="📄", layout="wide")




st.title("PDF Question Answer Tool")
st.markdown("### Upload a PDF and ask questions related to the content!")




from langchain.embeddings import HuggingFaceEmbeddings

embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
file_path="faiss_store_openai.pkl"


# File uploader for PDF documents
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

main_placeholder=st.empty()



if uploaded_file is not None :
    main_placeholder.text("Document Uploaded......✅✅✅")
    reader=PdfReader(uploaded_file)
    text=""
    for page in reader.pages:
        text=text+page.extract_text()+'\n'

    print(text)

    # split data
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n',"\n",'.',','],
        chunk_size=1000
    )

    documents = [Document(page_content=text,metadata={"source": uploaded_file.name})]
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs=text_splitter.split_documents(documents)
    vector_store_openai=FAISS.from_documents(docs,embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(0.2)

    with open(file_path,"wb") as f:
        pickle.dump(vector_store_openai,f)



query = st.text_input("Ask a Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore=pickle.load(f)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
            result = chain({"question" : query}, return_only_outputs=True)
            # {"answer":"","sources":[]}
            st.header("Answer")
            # st.subheader("Answer")
            st.subheader(result["answer"])


            # display sources if avialbele
            sources=result.get("sources","")
            if sources:
                st.subheader("Sources:")

                source_list=sources.split("\n")
                for source in source_list:
                    st.write(source)
