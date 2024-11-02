import os
import sys

from langchain_community.llms import Ollama
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from datetime import datetime

path = "R:\Manuais"                                                                                                     # Directory of the archive
text_loader_kwargs={'autodetect_encoding': True}                                                                          # Config the autodetect encoding
llm = Ollama(                                                                                                           # Settings of llm parameters
    #host = "10.21.0.220:11434",                                                                                        # host if default could be removed
    model="gemma2:2b",                                                                                                  # Choose of the Model
    temperature=0.9                                                                                                     # Creativity of the output result model
)

print("Start Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))                                                  # Print the date and time of Starting Loading Files
loader = DirectoryLoader(                                                                                               # Settings of loader set to DirectoryLoader
    path,                                                                                                               # Load the path previously set
    #glob="**/*.txt",                                                                                                   # Choose with type of file and file that will be used
    #loader_cls= TextLoader,                                                                                            # Help manage errors due to variations in file encodings.
    #loader_cls= PythonLoader,                                                                                          # Add PythonLoader if you are going to load Python source code files
    loader_kwargs = text_loader_kwargs,                                                                                 # Autodetect text encoding
    #silent_errors=True,                                                                                                # Skip the files which could not be loaded and continue the load process
    use_multithreading=True,                                                                                            # Default uses 1 thread to use multithread flag to True
    show_progress=True                                                                                                  # Show bar progress (install the tqdm library (e.g. pip install tqdm))
)

documents = loader.load()                                                                                               # Set documents to total of loaded files
print("End Loading =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))                                                    # Print the date and time of Ending Loading Files
len(documents)                                                                                                          # Show total of archives founded

print("Start Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))                              # Print the date and time of Splitting Storing Retriever of loaded files
text_splitter = CharacterTextSplitter(                                                                                  # Settings of text_splitter set to CharacterTextSplitter
    separator = r"\n",                                                                                                  # If the file has separator content text add the separator used separators=[" ", ",", "\n"]
    chunk_size = 1024,                                                                                                  # Maximum number of characters that chunk can contain
    chunk_overlap = 100,                                                                                                # Number of characters that should overlap between two adjacent chunks
    length_function=len,
    add_start_index=True
)

text_chunks = text_splitter.split_documents(documents)
print(f"Split {len(documents)} documents into {len(text_chunks)} chunks.")

embeddings = HuggingFaceEmbeddings()

knowledge_base = FAISS.from_documents(
    text_chunks,
    embeddings
)
print("End Splitting-Storing-Retriever =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

print("Start Condense Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever = knowledge_base.as_retriever()
)
print("End Condense Chaining =", datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

question = "Quantos monitores podem ser conectados a Central?"
response = qa_chain.invoke({"query": question})
print(response["result"])