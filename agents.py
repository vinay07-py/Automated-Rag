# src/agents.py
from crewai import Agent, LLM
import os

# Connect to Local Ollama Instance
local_llm = LLM(
    model="ollama/llama3.1",
    base_url="http://localhost:11434"
)

class DataAnalystCrew:
    def data_collection_agent(self):
        return Agent(
            role='Data Collection Specialist',
            goal='Ingest and profile incoming raw data files accurately.',
            backstory="You are an expert data engineer. You inspect file schemas, look at structures, detect delimiters, and assess structural profiles.",
            llm=local_llm,
            verbose=True
        )

    def data_cleaning_agent(self):
        return Agent(
            role='Data Cleaning Specialist',
            goal='Identify and correct missing data, anomalies, mismatched types, and outliers.',
            backstory="You maintain extreme data integrity. You plan strategies for filling null spaces, stripping noise, and fixing bad rows.",
            llm=local_llm,
            verbose=True
        )

    def feature_engineering_agent(self):
        return Agent(
            role='Feature Engineering Expert',
            goal='Formulate meaningful numerical transformations and extract maximum statistical signal.',
            backstory="You translate business context into matrices. You recommend encoding profiles, scaling methods, and high-value interaction variables.",
            llm=local_llm,
            verbose=True
        )

    def evaluation_agent(self):
        return Agent(
            role='Model Evaluation Critic',
            goal='Analyze data partitions and historical runtime performance metrics.',
            backstory="You are an unbiased auditor. You inspect validation matrices, overfitting concerns, accuracy scores, and performance targets.",
            llm=local_llm,
            verbose=True
        )

    def visualization_agent(self):
        return Agent(
            role='Data Storyteller and Visualizer',
            goal='Synthesize raw engineering actions into high-impact markdown summaries and strategic business intelligence charts.',
            backstory="You bridge the gap between technical operations and executive dashboards. You craft sleek, clear tactical summary reports.",
            llm=local_llm,
            verbose=True
        )