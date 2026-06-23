from langchain_google_genai import ChatGoogleGenerativeAI
from  dotenv import load_dotenv
import os 
load_dotenv()
print(os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

result = llm.invoke("what is the area of india?")
print(result)