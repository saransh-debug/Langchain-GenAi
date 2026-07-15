#helps picking up those results which are relevent but maintains diversity in the results , so that each result is unique , from each other. ALso known as MMR.   
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv 
from langchain_core.documents import Document

from langchain_core.documents import Document

load_dotenv()


docs = [
    Document(
        page_content="""
Glaciers are massive bodies of ice that move slowly over land.
They form from accumulated snow compressed over thousands of years.
Glaciers store most of the world's freshwater and shape mountain landscapes.
""",
        metadata={"topic": "glaciers", "id": 1}
    ),

    Document(
        page_content="""
The Himalayas contain some of the world's largest glaciers outside the polar regions.
Many major rivers in Asia originate from Himalayan glaciers.
Climate change is causing these glaciers to retreat at an alarming rate.
""",
        metadata={"topic": "glaciers", "id": 2}
    ),

    Document(
        page_content="""
Nature provides clean air, fresh water, and habitats for countless species.
Healthy ecosystems are essential for human survival and biodiversity.
Protecting nature helps maintain ecological balance for future generations.
""",
        metadata={"topic": "nature", "id": 3}
    ),

    Document(
        page_content="""
Forests are often called the lungs of the Earth because they absorb carbon dioxide.
They provide shelter to millions of plants and animals.
Conserving forests is vital for combating climate change.
""",
        metadata={"topic": "nature", "id": 4}
    ),

    Document(
        page_content="""
Hollywood is the center of the American film industry.
It is home to famous studios, actors, and filmmakers.
Many blockbuster movies are produced in Hollywood every year.
""",
        metadata={"topic": "hollywood", "id": 5}
    ),

    Document(
        page_content="""
The Hollywood Walk of Fame honors thousands of celebrities.
Millions of tourists visit Hollywood annually.
The Academy Awards celebrate excellence in filmmaking.
""",
        metadata={"topic": "hollywood", "id": 6}
    ),

    Document(
        page_content="""
Wildlife includes animals, birds, insects, and marine life living in natural habitats.
Conservation efforts help protect endangered species from extinction.
National parks play an important role in preserving wildlife.
""",
        metadata={"topic": "wildlife", "id": 7}
    ),

    Document(
        page_content="""
Oceans cover more than 70 percent of Earth's surface.
They regulate global climate and support diverse marine ecosystems.
Reducing plastic pollution is essential for protecting ocean life.
""",
        metadata={"topic": "oceans", "id": 8}
    ),

    Document(
        page_content="""
Mountains are formed through tectonic plate movements over millions of years.
They influence weather patterns and provide freshwater resources.
Many rare plants and animals thrive in mountainous regions.
""",
        metadata={"topic": "mountains", "id": 9}
    ),

    Document(
        page_content="""
Rainforests are among the most biodiverse ecosystems on Earth.
They produce oxygen and store vast amounts of carbon.
Protecting rainforests helps combat global warming and species loss.
""",
        metadata={"topic": "rainforest", "id": 10}
    ),
]



embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


vector_Store = FAISS.from_documents(
    embedding=embeddings , 
    documents=docs
)

retriever = vector_Store.as_retriever(
    search_type = "mmr" , 
    search_kwargs={"k":3 , "lambda_mult":0.2} # here lambda_mult means the diversity in the result if we give 0 , then the max diversity will be provided in the output , and if we give it as 1 then it performs the simple similarity search .
)

query = "how we can protect our nature "

result = retriever.invoke(query)

for i in result:
    print(i.metadata)
    print(i.page_content)