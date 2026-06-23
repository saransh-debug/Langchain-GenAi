from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    dimensions=32
)
document = [
    "my name is saransh",
    "my last name is bhardwaj"
]

result = embedding.embed_documents(document)
print(result)