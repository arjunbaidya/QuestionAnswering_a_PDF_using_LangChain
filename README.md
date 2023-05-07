# PDF Question-Answering app using LangChain, OpenAI, Pinecone and Streamlit

Note: Two versions have been created. First one uses FAISS for storing embeddings in memory. Second one used Pinecone for storing embeddings.

Upload your own PDF document to the app and ask questions to get answers from the PDF contents.

Here is a snapshot of the app. A PDF file is given to the app, then the app searches the contents of the PDF document using Semantic Search with OpenAI embeddings, then passes over the most relevant smaller pieces of information from the document to the OpenAI model using LangChain to formulate a cogent answer to the question and then displays the answer on the app for the user. 

![image](https://user-images.githubusercontent.com/109064198/236665163-e1f760e7-ef29-4623-ae3c-62bba390e6c8.png)
