from langchain_community.retrievers import WikipediaRetriever

query ="Kylian Mbappe" # spelling matter , if you didnt spelled the persons name here ,  you wont get the output else it will just print the  [] .

ret = WikipediaRetriever(top_k_results=1 , lang="en")

result = ret.invoke(query)

print(type(result) , len(result))

for i , val in enumerate(result):
    print(val.page_content)