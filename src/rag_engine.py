import yfinance as yf
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests
import json
import uuid
import os

from duckduckgo_search import DDGS

class StockRAGEngine:
    def __init__(self, collection_name="stock_news_v1"):
        """
        Initializes the RAG Engine with ChromaDB and SentenceTransformer.
        """
        self.chroma_client = chromadb.Client()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        self.ollama_base_url = "http://localhost:11434/api/generate"
        self.model_name = "phi3" # Switched to Phi-3 for better performance on 6GB VRAM

    def fetch_news_ddg(self, ticker):
        """
        Fetches news using DuckDuckGo.
        """
        try:
            results = []
            # Clean ticker for search (e.g., "RELIANCE.NS" -> "RELIANCE stock news")
            search_term = f"{ticker.split('.')[0]} stock news"
            
            with DDGS() as ddgs:
                ddgs_news = ddgs.news(search_term, region="in-en", safesearch="off", max_results=10)
                for item in ddgs_news:
                    results.append({
                        "id": str(uuid.uuid4()),
                        "title": item.get('title'),
                        "link": item.get('url'),
                        "published": item.get('date'),
                        "summary": item.get('body'),
                        "ticker": ticker,
                        "source": "DuckDuckGo"
                    })
            return results
        except Exception as e:
            print(f"Error fetching DDG news for {ticker}: {e}")
            return []

    def fetch_news(self, ticker):
        """
        Fetches news from Yahoo Finance.
        Handles both standard and nested Indian stock news structure.
        """
        try:
            full_ticker = ticker
            # Ensure .NS or .BO for Indian stocks if not present, though usually passed correctly
            yf_ticker = yf.Ticker(full_ticker)
            news_items = yf_ticker.news
            
            processed_news = []
            
            for item in news_items:
                # Handle nested structure (common in Indian stocks via YF)
                content = item.get('content', item) 
                
                title = content.get('title')
                link = content.get('clickThroughUrl') or content.get('link')
                pubDate = content.get('pubDate') or item.get('providerPublishTime')
                summary = content.get('summary')
                
                if title:
                    # Handle link being a dict (common in YF)
                    raw_link = content.get('clickThroughUrl') or content.get('link')
                    if isinstance(raw_link, dict):
                        link = raw_link.get('url')
                    else:
                        link = raw_link
                        
                    processed_news.append({
                        "id": str(uuid.uuid4()),
                        "title": title,
                        "link": str(link) if link else "N/A",
                        "published": str(pubDate),
                        "summary": summary,
                        "ticker": ticker,
                        "source": "YahooFinance"
                    })
            
            return processed_news
            
        except Exception as e:
            print(f"Error fetching YF news for {ticker}: {e}")
            return []

    def ingest_data(self, ticker):
        """
        Fetches news from multiple sources, cleans old data, generates embeddings, and stores in VectorDB.
        """
        # 1. Fetch from all sources
        yf_news = self.fetch_news(ticker)
        ddg_news = self.fetch_news_ddg(ticker)
        
        all_news = yf_news + ddg_news
        
        if not all_news:
            return "No news found from any source."

        # 2. Cleanup Old Data (Optimization)
        # Delete existing entries for this ticker to save space and keep context fresh
        try:
            self.collection.delete(where={"ticker": ticker})
        except:
            pass # Collection might be empty or delete might fail safely

        documents = []
        ids = []
        metadatas = []

        for item in all_news:
            # Create a rich text representation for the embedding
            # Include Source in the text to help the LLM cite it
            text_chunk = f"[{item['source']}] Title: {item['title']}\nSummary: {item['summary']}\nDate: {item['published']}"
            
            documents.append(text_chunk)
            ids.append(item['id'])
            metadatas.append({"ticker": ticker, "link": item['link'], "source": item['source']})

        # 3. Generate Embeddings & Store
        embeddings = self.embedding_model.encode(documents).tolist()

        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return f"Refreshed knowledge base! Ingested {len(documents)} articles ({len(yf_news)} via Yahoo, {len(ddg_news)} via DuckDuckGo) for {ticker}."

    def retrieve_context(self, query, ticker, n_results=5):
        """
        Queries the VectorDB for relevant news chunks.
        Increased n_results since we have more data now.
        
        Process:
        1. Convert user query to vector (embedding).
        2. Perform cosine similarity search in ChromaDB.
        3. Filter results to match the specific stock ticker.
        """
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where={"ticker": ticker} # Filter by current ticker
        )
        
        return results['documents'][0] if results['documents'] else []

    def generate_response(self, query, ticker):
        """
        End-to-end RAG output generation using Ollama.
        
        Steps:
        1. RETRIEVE: Get top-k relevant news snippets from VectorDB.
        2. AUGMENT: Inject these snippets into a prompt template.
        3. GENERATE: Send the augmented prompt to the local LLM (Phi-3).
        """
        # 1. Retrieve
        context_list = self.retrieve_context(query, ticker)
        context_str = "\n\n".join(context_list)
        
        if not context_str:
            return "No relevant news found to answer this question."

        # 2. Construct Prompt
        # We explicitly ask the model to act as a financial assistant and cite sources.
        prompt = f"""
        You are a financial analyst assistant. Use the following news snippets to answer the user's question about {ticker}.
        Each snippet starts with [Source]. Please cite sources where possible.
        
        --- RELEVANT NEWS ---
        {context_str}
        ---------------------
        
        Question: {query}
        
        Answer (be concise, professional, and cite sources):
        """

        # 3. Call Ollama
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 2048 # Reduced from 4096 to save VRAM on 6GB cards
                }
            }
            response = requests.post(self.ollama_base_url, json=payload)
            if response.status_code == 200:
                return response.json().get('response', "Error: Empty response from Ollama.")
            else:
                return f"Error communicating with Ollama: {response.status_code}"
                
        except Exception as e:
            return f"Ollama Connection Error: {e}. Is 'ollama serve' running?"

# Simple test if run directly
if __name__ == "__main__":
    engine = StockRAGEngine()
    t = "RELIANCE.NS"
    print(f"Ingesting {t}...")
    print(engine.ingest_data(t))
    
    q = "What is the recent news about Reliance?"
    print(f"Query: {q}")
    print(engine.generate_response(q, t))
