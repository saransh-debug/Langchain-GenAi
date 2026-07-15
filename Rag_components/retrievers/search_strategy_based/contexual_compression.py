# in this retriver , we have the feature to get the desired and neccessary output of the avaiable data in the db , there is a base retriver which fetches the top k similar results from the vector db , and then there we have a compressor which have the job of sending those outputs of the LLM and the llm cuts down the extra non-required info out of it and just gives out the required info , and the retriever retrives it and sends it to us as the output . 
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from dotenv import load_dotenv 
from langchain_core.documents import Document

from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever 
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
load_dotenv()




embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


docs = [
    Document(
        page_content="""
The Grand Canyon is one of the most famous natural landmarks in the United States.
Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen.
The Colorado River has carved the canyon over millions of years.
"""
    ),

    Document(
        page_content="""
Mount Everest is the highest mountain above sea level and attracts climbers from around the world.
Photosynthesis takes place mainly in the chloroplasts of plant cells, where chlorophyll captures sunlight.
Snowfall and glaciers shape many high-altitude mountain landscapes.
"""
    ),

    Document(
        page_content="""
The Pacific Ocean is the largest ocean on Earth and is home to diverse marine life.
Plants use photosynthesis to produce their own food, releasing oxygen as a by-product that supports life on Earth.
Ocean currents influence weather patterns across the globe.
"""
    ),

    Document(
        page_content="""
Modern cities rely on efficient transportation systems and renewable energy sources.
During photosynthesis, plants absorb carbon dioxide from the atmosphere and use sunlight to synthesize glucose.
Public parks improve the quality of urban life and increase green spaces.
"""
    ),

    Document(
        page_content="""
Astronauts aboard the International Space Station conduct experiments in microgravity.
Photosynthesis is essential for maintaining Earth's oxygen levels and forms the foundation of most food chains.
Space exploration continues to expand our understanding of the universe.
"""
    )
]

vector_Store = FAISS.from_documents(docs , embeddings)

base_Ret  = vector_Store.as_retriever(search_kwargs={"k":3})

simple_res= base_Ret.invoke("what is photosynthesis?")

compressor = LLMChainExtractor.from_llm(llm)

compression_retriever = ContextualCompressionRetriever(base_compressor=compressor , 
                                                       base_retriever=base_Ret)

result = compression_retriever.invoke("what is photosynthesis?")



for i ,doc in enumerate(simple_res):
    print("------------------------without compression output----------------------")
    print(f"{i+1}---{doc.page_content}")


    
for i , doc in enumerate(result):
    print("------------------------with compression output----------------------")
    print(f"{i+1}---{doc.page_content}")