# src/tasks.py
from crewai import Task
from src.agents import data_collector, data_trainer, feature_engineer, ml_trainer, evaluator, visualization_agent

task_collect = Task(
    description="Load the raw dataset from '{dataset_path}' using a pandas execution context. Output columns and data structural type maps.",
    expected_output="A structured column blueprint mapping the verified input shape.",
    agent=data_collector
)

task_train_prep = Task(
    description="Analyze missing values, perform median/mode cleanups, split into an 80/20 partition model, and save outputs to data/processed/train.csv and data/processed/test.csv.",
    expected_output="Clean partitioned file instances verified on local storage volumes.",
    agent=data_trainer
)

task_feature_eng = Task(
    description="Extract structural correlations from the processed dataset splits. Encode string categories programmatically into analytical markers.",
    expected_output="Final engineered column sets ready for model ingestion tasks.",
    agent=feature_engineer
)

task_ml_train = Task(
    description="Read data/processed/train.csv and fit an algorithm blueprint (Random Forest/XGBoost). Export the serialized model block to the models/ folder.",
    expected_output="A generated model binary saved locally.",
    agent=ml_trainer
)

task_evaluate = Task(
    description="Inference against data/processed/test.csv using the exported binary. Calculate classification precision, recall metrics, and validation loss arrays.",
    expected_output="Complete system matrix detailing mathematical evaluations.",
    agent=evaluator
)

task_visualize = Task(
    description="Synthesize all structural outputs, performance scores, and data transformations into a complete, clean executive Markdown summary report.",
    expected_output="A robust, beautifully structured Markdown presentation document.",
    agent=visualization_agent
)