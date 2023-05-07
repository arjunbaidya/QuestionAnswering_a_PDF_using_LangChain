from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()

    # Creating the Streamlit app interface
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 🧐")
    pdf = st.file_uploader("Upload your PDF file", type="pdf")
    
    # Reading the file contents
    if pdf is not None:

        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()           

        # Splitting up the text into smaller chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text)

        # OpenAI embeddings
        embeddings = OpenAIEmbeddings()
        
        # Create vector knowledge base with FAISS
        vector_base = FAISS.from_texts(chunks, embeddings)

        # Take a question from the user
        query = st.text_input("Ask a question from the PDF contents:")
                
        if query:
            # Get the documents that are similar to the question asked
            similar_docs = vector_base.similarity_search(query, k=3)
            
            # Load the QA chain and run it to get the response, also track spending on OpenAI 
            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=similar_docs, question=query)
                print(cb)

            # Post the answer on the app
            st.write(response)

if __name__=='__main__':
    main()

