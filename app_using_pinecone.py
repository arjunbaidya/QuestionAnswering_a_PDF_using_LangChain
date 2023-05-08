# This implementation uses Pinecone to store the vectors 

from dotenv import load_dotenv, dotenv_values
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import pinecone

def main():
    load_dotenv()
    config = dotenv_values(".env")
    PINECONE_API_KEY = config["PINECONE_API_KEY"]
    PINECONE_API_ENV = config["PINECONE_API_ENV"]

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
        
        # Create vector knowledge base with Pinecone
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_API_ENV
        )
        index_name = "langchain"

        vector_base = Pinecone.from_texts(
            texts=chunks,
            embedding=embeddings,
            index_name=index_name
        )

        # Take a question from the user
        query = st.text_input("Ask a question from the PDF contents:")
                
        if query:
            # Get the documents that are similar to the question asked
            similar_docs = vector_base.similarity_search(query, k=3, include_metadata=True)
            
            # Load the QA chain and run it to get the response, also track spending on OpenAI 
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=similar_docs, question=query)
                print(cb)

            # Post the answer to the user on the app
            st.write(response)

if __name__=='__main__':
    main()

