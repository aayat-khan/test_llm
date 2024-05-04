import os
import pickle
import langchain
langchain.debug=False
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
DEBUG=True
llm = HuggingFaceHub(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    huggingfacehub_api_token='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',
    model_kwargs={"temperature":0.8, "max_length":1000}
)
if DEBUG : print("llm loaded")
import nest_asyncio

nest_asyncio.apply()
loader = MongodbLoader(
    connection_string="mongodb+srv://khanaayat606:Aayat%40123@cluster0.jb7h0fj.mongodb.net/?retryWrites=true&w=majority",
    db_name="cluster0",
    collection_name="my_collection",
    field_names=["genre", "title"],
)
if DEBUG : print("mongo loaded")

data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# As data is of type documents we can directly use split_documents over split_text in order to get the chunks.
docs = text_splitter.split_documents(data)
if DEBUG : print("text splitting done")
# Create the embeddings of the chunks using openAIEmbeddings
embeddings = HuggingFaceEmbeddings()
if DEBUG : print("embeddings initialised")
# Pass the documents and embeddings inorder to create FAISS vector index
vectorindex_openai = FAISS.from_documents(docs, embeddings)
if DEBUG : print("FAISS done")
# Storing vector index create in local
file_path="vector_index.pkl"
with open(file_path, "wb") as f:
    pickle.dump(vectorindex_openai, f)

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorIndex = pickle.load(f)
        # Storing vector index create in local 
if DEBUG : print("vector indexing done")
chain = RetrievalQA.from_chain_type(llm=llm,
                                 retriever=vectorIndex.as_retriever(),
                                 return_source_documents=True)
if DEBUG : print("chain created")
from langchain.chains.conversation.memory import ConversationBufferMemory

chain = RetrievalQA.from_chain_type(llm=llm,
                                 retriever=vectorIndex.as_retriever(),#buffer
                                 return_source_documents=True,
                                 chain_type="stuff",
                                 verbose=False,
    chain_type_kwargs={
        "verbose":False,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    })
if DEBUG : print("model done")


list_of_questions = [     "Do you like horror or action movies?",     
                     "Family or Romance, what do you prefer?",    
                     "Comedy or Adventure, which one would you pick?"  ]


choices = [] 
for question in list_of_questions:     
    user_choice = input(f'{question}')     
    choices.append(user_choice) 
    
if DEBUG : print("append")    

prompt = f'Recommend a movie which is {choices[0]}, {choices[1]}, {choices[2]}' 
result = chain({"query": prompt}, return_only_outputs=True) 
text = result['result']
 # Extract the text after "Answer:"
answer_start = text.find("Answer:")  
 # until a blank line is found
answer_text = text[answer_start:text.find("\n\n", answer_start)] 
# Print the extracted text
print(answer_text)
