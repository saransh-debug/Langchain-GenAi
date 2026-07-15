from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()
docs = [
     Document(
        page_content="Lionel Messi is an Argentine footballer who won the 2022 FIFA World Cup and has won multiple Ballon d'Or awards.",
        metadata={"name": "Lionel Messi", "category": "Sports"}
    ),
    Document(
        page_content="Marie Curie was a physicist and chemist who discovered radium and polonium and won two Nobel Prizes.",
        metadata={"name": "Marie Curie", "category": "Science"}
    ),
    Document(
        page_content="Elon Musk is the CEO of Tesla and SpaceX and has contributed to electric vehicles, reusable rockets, and AI.",
        metadata={"name": "Elon Musk", "category": "Business"}
    ),
    Document(
        page_content="Mahatma Gandhi led India's independence movement through non-violent resistance and inspired global civil rights movements.",
        metadata={"name": "Mahatma Gandhi", "category": "History"}
    ),
    Document(
        page_content="Taylor Swift is a Grammy-winning singer-songwriter known for her successful albums and the record-breaking Eras Tour.",
        metadata={"name": "Taylor Swift", "category": "Music"}
    ),
]

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

vector_Store = FAISS.from_documents(
    embedding= embedding , 
    documents=docs , 
    
)

ret = vector_Store.as_retriever(search_kwargs={"k":1})

query = " gandhi"

result = ret.invoke(query)

print(result[0].metadata)
print(result[0].page_content)

