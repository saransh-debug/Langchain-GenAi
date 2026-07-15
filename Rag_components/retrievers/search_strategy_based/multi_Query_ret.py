# in multi query retriever, we will use the search strategy to retrieve the documents for each query and then combine the results. The search strategy can be any of the strategies defined in the search_strategy_based module, such as BM25, TF-IDF, or a custom strategy.
# it takes query as an input and breaks it into multiple sub-queries, then retrieves documents for each sub-query using the specified search strategy, and finally combines the results into a single list of documents.

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI
from dotenv import load_dotenv 
from langchain_core.documents import Document

from langchain_core.documents import Document
from langchain_classic.retrievers import MultiQueryRetriever
load_dotenv()


docs = [
    Document(
        page_content="""
A balanced diet provides essential nutrients for the body's growth and repair.
Eating fruits, vegetables, whole grains, and proteins supports overall health.
Limiting processed foods can reduce the risk of chronic diseases.
""",
        metadata={"topic": "Nutrition", "id": 1}
    ),

    Document(
        page_content="""
Regular exercise strengthens the heart, muscles, and bones.
Adults should aim for at least 150 minutes of moderate physical activity each week.
Exercise also improves mental health and reduces stress.
""",
        metadata={"topic": "Exercise", "id": 2}
    ),

    Document(
        page_content="""
Sleep is essential for memory, concentration, and physical recovery.
Most adults require seven to nine hours of sleep each night.
Poor sleep can increase the risk of obesity and heart disease.
""",
        metadata={"topic": "Sleep", "id": 3}
    ),

    Document(
        page_content="""
Drinking enough water helps regulate body temperature and transport nutrients.
Dehydration can cause fatigue, headaches, and reduced concentration.
Water is essential for healthy kidney function.
""",
        metadata={"topic": "Hydration", "id": 4}
    ),

    Document(
        page_content="""
Mental health includes emotional, psychological, and social well-being.
Managing stress through relaxation and mindfulness improves quality of life.
Seeking professional help is important when mental health challenges persist.
""",
        metadata={"topic": "Mental Health", "id": 5}
    ),

    Document(
        page_content="""
Vaccines help protect people from infectious diseases.
They train the immune system to recognize and fight harmful pathogens.
Vaccination has significantly reduced many life-threatening illnesses.
""",
        metadata={"topic": "Vaccination", "id": 6}
    ),

    Document(
        page_content="""
Good hygiene helps prevent the spread of bacteria and viruses.
Regular handwashing is one of the most effective ways to avoid infections.
Maintaining personal cleanliness supports overall health.
""",
        metadata={"topic": "Hygiene", "id": 7}
    ),

    Document(
        page_content="""
The immune system protects the body against harmful microorganisms.
A healthy lifestyle supports proper immune function.
Poor nutrition and lack of sleep can weaken immunity.
""",
        metadata={"topic": "Immune System", "id": 8}
    ),

    Document(
        page_content="""
Heart health depends on regular exercise, healthy eating, and avoiding smoking.
High blood pressure and high cholesterol increase cardiovascular risk.
Routine health checkups can help detect problems early.
""",
        metadata={"topic": "Heart Health", "id": 9}
    ),

    Document(
        page_content="""
Diabetes is a condition that affects how the body regulates blood sugar.
Healthy eating, physical activity, and medication can help manage diabetes.
Regular monitoring of blood glucose levels is important for long-term health.
""",
        metadata={"topic": "Diabetes", "id": 10}
    ),
]

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_Store = FAISS.from_documents(
    documents=docs ,  
    embedding=embeddings
)

# simple retreiver 
 
simple_Ret = vector_Store.as_retriever(
    search_type="mmr",
    search_kwargs={"k":1}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

# Multi_query retriver 

multi_query = MultiQueryRetriever.from_llm(
     retriever=simple_Ret,
    llm=llm
)


result = multi_query.invoke(
    "How can I keep my heart healthy?"
)

res_sim = simple_Ret.invoke("How can I keep my heart healthy?")
for i in res_sim:
    print(f"simple_ret answers : {i.page_content}")
for i,doc in enumerate (result) :
    print(f"{i+1}________{doc.page_content}")