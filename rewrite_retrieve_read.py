import os

from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in your environment or a .env file."
    )

#  Sample corpus 
texts = [
    "Apple MacBook Air M2 is lightweight and suitable for developers, with long battery life.",
    "Dell XPS 13 is a high-end Windows laptop with good performance for programming.",
    "Lenovo ThinkPad X1 Carbon is durable and has a comfortable keyboard, great for coding.",
    "Asus ZenBook 14 offers strong performance at a lower price point, ideal for budget-conscious developers."
]

# Create embeddings and vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(texts, embedding=embeddings)

llm = ChatOpenAI(model="gpt-4o-mini")

#  Step 1: Rewrite 
rewrite_prompt = PromptTemplate.from_template("""
Rewrite the user's query to be a clear, precise search query suitable for document retrieval.
Original query: "{query}"
""")

user_query = "cheap laptop coding"
response = llm.invoke(rewrite_prompt.format(query=user_query))

# Handle potential list-like content payloads defensively
def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and "text" in p:
                parts.append(p.get("text", ""))
            else:
                parts.append(str(p))
        return "\n".join([s for s in parts if s])
    return str(content)

rewritten_query = _content_to_text(response.content)
print("Rewritten Query:\n", rewritten_query)

#  Step 2: Retrieve 
retrieved_docs = vectorstore.similarity_search(rewritten_query, k=2)
print("\nRetrieved Documents:")
for doc in retrieved_docs:
    print("-", doc.page_content)

#  Step 3: Read/Answer 
read_prompt = PromptTemplate.from_template("""
You are given the following context from documents:
{context}

Answer the user's original question:
"{query}"
""")

context_text = "\n".join([d.page_content for d in retrieved_docs])
final_answer = llm.invoke(read_prompt.format(context=context_text, query=user_query))
print("\nFinal Answer:\n", final_answer.content)
