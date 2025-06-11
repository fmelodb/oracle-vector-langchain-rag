import os
os.environ["USER_AGENT"] = "RAG-demo"

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

from langchain_community.vectorstores.oraclevs import OracleVS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv
import oracledb


# load environment variables
load_dotenv() 

def connect_database():
    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    dsn = os.getenv("DB_URL")

    try:
        connection = oracledb.connect(user=username, password=password, dsn=dsn)
        print("Connection successful!")
    except Exception as e:
        print("Connection failed!")
    
    return connection


# 1. Split data into chunks
urls = [
    "https://en.wikipedia.org/wiki/GeForce_RTX_50_series",
    "https://en.wikipedia.org/wiki/GeForce_RTX_40_series",
    "https://en.wikipedia.org/wiki/GeForce_RTX_30_series"
]

# embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")  

# model service - Mistral-7B-Instruct-v0.3.Q4_K_M
model_service = "http://localhost:51524/v1/"

# vector database connection
connection = connect_database()

# drop table if exists
print("Drop table\n")
with connection.cursor() as cursor:
    sql = "drop table if exists Documents_COSINE"
    try:
        cursor.execute(sql)
    except oracledb.DatabaseError as e:
        if e.args[0].code != 942:
            raise

# get url and convert to embeddings
print("Convert web sites into embeddings\n")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(docs_list)

# 2. Convert documents to Embeddings and store them
print("Save embeddings into the vector database\n")
try:
    vector_store = OracleVS.from_documents(
        chunks,
        embedding_model,
        client=connection,
        table_name="Documents_COSINE",
        distance_strategy=DistanceStrategy.COSINE,
    )
except Exception as e:
    raise

llm = OpenAI(base_url=model_service,
             api_key="sk-no-key-required",
             streaming=True)

prompt_template = "the most powerful graphics card model ever made by NVidia in 2025?"

print("Before RAG")
before_rag_template = "What is {topic}"
prompt = PromptTemplate(
        input_variables=["topic"],
        template= before_rag_template
    )

chain = prompt | llm
print(chain.invoke({'topic': prompt_template}))
    
print("\n########\nAfter RAG")
after_rag_template = "Answer the question based only in the following context: {context} Question: {question}"
after_rag_prompt = PromptTemplate.from_template(after_rag_template)

result_data = vector_store.similarity_search("most powerful graphics card model by NVidia in 2025?", k=3)
all_results = " ".join([d.page_content for d in result_data])

prompt = PromptTemplate(
        input_variables=["context", "question"],
        template= after_rag_template
    )
    
chain = prompt | llm
response = chain.invoke({'question': "What is the most powerful graphics card model ever made by NVidia in 2025?", 'context' : all_results}) 
print(response)
