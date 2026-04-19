import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Data to ingest - Updated with 2024-2025 Bangalore Market Intelligence
MARKET_DATA = [
    {
        "location": "Whitefield",
        "content": "Whitefield Market 2025: Transitioning from a purely IT hub to a premium residential destination. The Purple Line metro completion in 2024 has led to a 15% increase in capital values. Micro-markets like Varthur and Gunjur are now the primary growth zones for luxury villa projects."
    },
    {
        "location": "Sarjapur Road",
        "content": "Sarjapur Road 2025 Outlook: High-density residential growth continues with 7-9% annual appreciation expected thru 2026. The Peripheral Ring Road (PRR) progress is the key long-term driver. Rental yields remain strong at 3.5-4.2% for 3BHK gated communities."
    },
    {
        "location": "Electronic City",
        "content": "Electronic City Insight 2024: Phase 1 remains a steady rental hub with 3% growth in 2024. Despite traffic concerns, demand for affordable and mid-range housing remains resilient due to the large workforce in the area."
    },
    {
        "location": "Hebbal",
        "content": "Hebbal & North Bangalore 2025: This remains the fastest-growing investment corridor due to proximity to Kempegowda International Airport and the upcoming Devanahalli Business Park. High-net-worth individuals are targeting premium high-rise projects here."
    },
    {
        "location": "Indiranagar & Koramangala",
        "content": "Premium Core 2025: These areas are seeing 10-15% annual rental yield increases in 2024-2025, driven by the expansion of Global Capability Centers (GCCs) and high-end retail developments. Supply remains extremely limited."
    },
    {
        "location": "General Regulation",
        "content": "Legal & RERA 2025: K-RERA strictly mandates project registration for any development > 500 sq meters. Properties with 'A-Khata' documentation command a 15-20% premium over B-Khata due to easier financing and clear titles."
    },
    {
        "location": "Financials",
        "content": "Stamp Duty 2025: Karnataka maintains a 5% stamp duty for properties above ₹45 Lakhs. Registration fees are 1% of the property value. Always factor in an additional 7-8% over the agreement value for total acquisition costs."
    }
]

def ingest():
    # Use localized HuggingFace embeddings (all-MiniLM-L6-v2 is small and fast)
    print("Loading HuggingFace embeddings (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    docs = [
        Document(page_content=item["content"], metadata={"location": item["location"]})
        for item in MARKET_DATA
    ]
    
    print(f"Ingesting {len(docs)} documents into ChromaDB...")
    
    # Save to disk (overwrite existing)
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    
    print("Ingestion complete. Vector store saved to ./chroma_db")

if __name__ == "__main__":
    ingest()
