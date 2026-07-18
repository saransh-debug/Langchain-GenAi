from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpointEmbeddings , HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda , RunnableParallel , RunnablePassthrough

load_dotenv()

query = input("enter the text here :")
# transcript solution -----------------------------------------------------------------------------------------------------
yt_api = YouTubeTranscriptApi()

res = yt_api.fetch(
    video_id="7ARBJQn6QkM" , 
    languages=['en']
)

transcript = " ".join(i['text'] for i in res.to_raw_data())
# print(transcript)

Splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000 , 
    chunk_overlap = 200
)

docs = Splitter.create_documents([transcript])


#-------------------------------------indexing----------------------------------------------------------------------
embeddings = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

vector_store = FAISS.from_documents(
    documents=docs ,  
    embedding=embeddings

)

retriver = vector_store.as_retriever(search_type ="mmr" , search_kwargs={"k":4 ,"lambda_mult":0.2})

# print(retriver.invoke("what are robots"))




def context_extractor(result):
    # print(result)
    context = "\n\n".join(doc.page_content for doc in result)
    return context



#-------------------------------------chaining-----------------------------------------------------------------
parallel_chain = RunnableParallel(
    question = RunnablePassthrough() , 
    context = retriver| RunnableLambda(context_extractor)
)

# print(parallel_chain.invoke("what are robots?"))


template = PromptTemplate(
    template="""
You are a YouTube video summarizer.

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)


chatmodel = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    
) 

llm = ChatHuggingFace(llm=chatmodel)

parser = StrOutputParser()

main_chain = parallel_chain | template | llm | parser



print(main_chain.invoke(query))



