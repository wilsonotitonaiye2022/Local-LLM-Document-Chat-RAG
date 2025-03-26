# Description: This is the main file that contains the Streamlit code for the AI Assistant chatbot that 
# answers questions based on the uploaded files.


# Importing the required libraries
import streamlit as st
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
import os
import system_prompting


# Chat UI title
st.title("Local AI Assistant for Document interaction :male-technologist:")
st.markdown(
    """
    This app was implemented uses :100:% open-source Large language Models (LLMs). It uses locally installed \
    LLM models on the Ollama framework. A couple of models was used in the implementation of this app; \
    [BGE-M3](https://ollama.com/library/bge-m3) was used for embedding the document while \
    [GEMMA 3 12B](https://ollama.com/library/gemma3) was used for retrival of responses.
    """
)


# Load the system prompt    
system_prompt = system_prompting.DOC_SYSTEM_PROMPT
# Initialize the LLMChain
llm = OllamaLLM(model="gemma3:12b")
ollama_embedding = OllamaEmbeddings(model="bge-m3")
system_template = SystemMessagePromptTemplate.from_template("{system_prompt}")
user_template = HumanMessagePromptTemplate.from_template("{user_prompt}")
template = ChatPromptTemplate.from_messages([system_template, user_template])
chain = LLMChain(llm=llm, prompt=template)

        
with st.sidebar:
    st.subheader(":bulb: About: ")
    st.info("This is a chatbot that answers questions based on the uploaded files. Please upload your files \
            to get started. The app uses a RAG implematation appoach. It uses the Ollama framework to generate \
            embeddings from uploaded document and run LLM locally.The app uses the langchain library to create a \
            conversational retrieval chain.")
    # Display the information about the file types supported
    uploaded_files = st.file_uploader("**Please upload your files. File type supported: PDF/DOCX/TXT :page_facing_up:**", 
                                      accept_multiple_files=True, type=None)
    
    st.info("Please refresh the browser if you decided to upload more files to reset the session", icon="🚨")
    #Add refresh button to clear the session
    if st.button("Refresh"):
        # Delete all the items in Session state
        for key in st.session_state.keys():
            del st.session_state[key]


# Check if files are uploaded
if uploaded_files:
    # Print the number of files to console
    print(f"Number of files uploaded: {len(uploaded_files)}")

    # Load the data and perform preprocessing only if it hasn't been loaded before
    if "processed_data" not in st.session_state:
        # Load the data from uploaded PDF files
        documents = []
        for uploaded_file in uploaded_files:
            # Get the full file path of the uploaded file
            file_path = os.path.join(os.getcwd(), uploaded_file.name)

            # Save the uploaded file to disk
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            # Use UnstructuredFileLoader to load the PDF file
            loader = UnstructuredFileLoader(file_path)
            loaded_documents = loader.load()
            print(f"Number of files loaded: {len(loaded_documents)}")

            # Extend the main documents list with the loaded documents
            documents.extend(loaded_documents)

        # Chunk the data, create embeddings, and save in vectorstore
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        document_chunks = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(document_chunks, ollama_embedding)


        # Store the processed data in session state for reuse
        st.session_state.processed_data = {
            "document_chunks": document_chunks,
            "vectorstore": vectorstore,
        }

        # Print the number of total chunks to console
        print(f"Number of total chunks: {len(document_chunks)}")

    else:
        # If the processed data is already available, retrieve it from session state
        document_chunks = st.session_state.processed_data["document_chunks"]
        vectorstore = st.session_state.processed_data["vectorstore"]

    # Initialize Langchain's QA Chain with the vectorstore
    qa = ConversationalRetrievalChain.from_llm(llm,vectorstore.as_retriever())


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask your questions?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query the assistant using the latest chat history
        result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in st.session_state.messages]})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            full_response = result["answer"]
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)    
        print(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.warning("Please upload your file/files to chat with AI Assitant.")
