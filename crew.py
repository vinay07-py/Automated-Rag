# src/crew.py
from crewai import Crew, Process
from src.agents import data_collector, data_trainer, feature_engineer, ml_trainer, evaluator, visualization_agent, local_llm

class AutonomousDataScienceCrew:
    def __init__(self):
        pass

    def run_pipeline(self, dataset_path):
        from src.tasks import task_collect, task_train_prep, task_feature_eng, task_ml_train, task_evaluate, task_visualize
        
        inputs = {'dataset_path': dataset_path}
        
        local_crew = Crew(
            agents=[
                data_collector, data_trainer, feature_engineer, 
                ml_trainer, evaluator, visualization_agent
            ],
            tasks=[
                task_collect, task_train_prep, task_feature_eng, 
                task_ml_train, task_evaluate, task_visualize
            ],
            process=Process.sequential,
            planning_llm=local_llm,
            output_log_file="logs/crew_execution.log" # Asynchronous safe file write targeting
        )
        
        return local_crew.kickoff(inputs=inputs)