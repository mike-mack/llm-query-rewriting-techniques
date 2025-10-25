"""
Multi-Query Retrieval demo using LangChain + OpenAI.

Changes from the original:
- Replace FAISS with Chroma (avoids faiss-cpu install issues on macOS/arm64).
- Use OpenAIEmbeddings from langchain_openai (current, non-deprecated import).
- Add a clear check for OPENAI_API_KEY.
"""

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


# Setup
texts = [
    "Paris is the capital of France and known for its art and fashion.",
    "Berlin is the capital of Germany and a major tech hub in Europe.",
    "Tokyo, Japan's capital, is famous for its technology and cuisine.",
    "Seoul, the capital of South Korea, is known for K-pop, cutting-edge electronics, and vibrant street food culture.",
    "Paris hosts the Eiffel Tower, one of the most visited monuments in the world."
]

# Create embeddings & Chroma vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(texts, embedding=embeddings)

llm = ChatOpenAI(model="gpt-4o-mini")

# Step 1: Multi-query generation
multiquery_prompt = PromptTemplate.from_template("""
You are helping improve information retrieval.
Given the question: "{query}"
Generate 5 alternative phrasings that might retrieve different relevant information.
Output each on a new line.
""")

user_query = "What city has the Eiffel Tower?"
response = llm.invoke(multiquery_prompt.format(query=user_query))

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

resp_text = _content_to_text(response.content)
alt_queries = [user_query] + [q.strip() for q in resp_text.splitlines() if q.strip()]

print("Generated Queries:")
for q in alt_queries:
    print(" -", q)

# Step 2: Retrieve for each query
retrieved_docs = []
for q in alt_queries:
    docs = vectorstore.similarity_search(q, k=2)
    retrieved_docs.extend(docs)

# Deduplicate results (based on page content)
unique_docs = list({d.page_content: d for d in retrieved_docs}.values())

print("\nRetrieved Documents:")
for d in unique_docs:
    print("-", d.page_content)

# Step 3: Combine & answer
context = "\n".join([d.page_content for d in unique_docs])
answer_prompt = PromptTemplate.from_template("""
Use the context below to answer the question.

Context:
{context}

Question:
{query}
""")

final_answer = llm.invoke(answer_prompt.format(context=context, query=user_query))
print("\nFinal Answer:\n", final_answer.content)
