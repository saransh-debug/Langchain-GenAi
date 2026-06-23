from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    dimensions=32
)

result = embedding.embed_query("my name is saransh")
print(result)