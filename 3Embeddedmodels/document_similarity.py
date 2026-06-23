from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()

emdeddings = GoogleGenerativeAIEmbeddings(
    model = "models/gemini-embedding-001",
    dimensions=300
)

document = [
  "Virat Kohli is one of the best batsmen in modern cricket.",
"Rohit Sharma holds the record for the highest individual score in ODI cricket.",
"MS Dhoni led India to victory in the 2011 Cricket World Cup.",
"Sachin Tendulkar is known as the God of Cricket.",
"Jasprit Bumrah is famous for his exceptional fast bowling."
]

query = "tell me about virat ."

doc_emb = emdeddings.embed_documents(
    document
)

query_embd = emdeddings.embed_query(query)

result = cosine_similarity([query_embd],doc_emb)[0]

index, score = (sorted(list(enumerate(result)),key=lambda x :x[1])[-1])

print(document[index])
