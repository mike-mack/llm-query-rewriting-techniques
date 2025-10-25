"""
Hypothetical Document Embeddings (HyDE) demo using LangChain + OpenAI.

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

# === Setup ===
texts = [
    "Paris is the capital of France, famous for art, cuisine, and the Eiffel Tower.",
    "Berlin is known for its startup culture and historical landmarks.",
    "Tokyo leads in robotics, innovation, and modern architecture.",
    "Seoul has become a hub of pop culture, with K-pop and advanced electronics."
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(texts, embedding=embeddings)

llm = ChatOpenAI(model="gpt-4o-mini")

# === HyDE Step 1: Generate a hypothetical answer ===
hyde_prompt = PromptTemplate.from_template("""
You are helping to improve information retrieval.
Given the question: "{query}"
Write a short paragraph that *plausibly answers* this question,
even if you are unsure. Keep it factual-sounding and informative.
""")

query = "Which city is famous for its robotics industry?"

response = llm.invoke(hyde_prompt.format(query=query))

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

hypothetical_doc = _content_to_text(response.content)
print("Hypothetical Document:\n", hypothetical_doc)

# === HyDE Step 2: Embed and retrieve ===
query_vector = embeddings.embed_query(hypothetical_doc)
results = vectorstore.similarity_search_by_vector(query_vector, k=2)

print("\nRetrieved Documents:")
for doc in results:
    print("-", doc.page_content)
