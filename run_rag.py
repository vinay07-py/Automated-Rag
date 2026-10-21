# run_rag.py
import sys
import os
import polars as pl
import json
import urllib.request


def query_local_ollama(prompt, model_name="llama3.1"):
    """Bypasses heavy dependencies using standard library HTTP requests to talk to local Ollama."""
    url = "http://localhost:11434/api/generate"
    data = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "No structural response returned from local LLM matrix context.")
    except Exception as e:
        return f"Ollama Connection Error: Ensure ollama serve is running locally. Error details: {str(e)}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Missing query parameter argument context loops.")
        sys.exit(1)

    query = sys.argv[1]
    sample_data_path = "data/processed/rag_context_sample.csv"
    report_path = "logs/final_report.md"

    # Collect data context footprints to feed inside LLM context window arrays safely
    context_footprint = ""
    if os.path.exists(sample_data_path):
        df = pl.read_csv(sample_data_path)
        context_footprint += f"Dataset Schema Dimensions: {df.height} rows mapped x {df.width} columns.\n"
        context_footprint += f"Available features structural names: {', '.join(df.columns)}\n"
        context_footprint += f"Sample Row Vector Profiles:\n{df.head(3)}\n"

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            context_footprint += f"\nCompiled Model Report Performance Data:\n{f.read()}"

    # Construct isolated RAG context engineering injection payload
    rag_prompt = f"""
You are the primary Data Intelligence RAG Core for the NEO-DS multi-agent platform.
Analyze the following compiled system context metrics and data blueprints below to answer the user request.

[DATA WORKSPACE METRIC BUFFER]
{context_footprint}

[USER INQUIRY TRANSACTION OBJECTIVE]
{query}

Provide a direct, authoritative analytical answer. Do not repeat the context values verbatim.
"""
    answer = query_local_ollama(rag_prompt)
    print(answer)