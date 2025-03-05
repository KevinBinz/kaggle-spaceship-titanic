import pandas as pd
import os
from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from experiment_recorder import ExperimentRecorder

def main():
    # Initialize components
    data_loader = DataLoader()
    feature_engineer = FeatureEngineer()
    model_trainer = ModelTrainer()
    experiment_recorder = ExperimentRecorder()

    # Load and preprocess data
    train_df, test_df = data_loader.load_data()
    train_df, test_df = feature_engineer.engineer_features(train_df), feature_engineer.engineer_features(test_df)

    # Initialize H2O and train model
    model_trainer.init_h2o()
    model_trainer.load_data(train_df, test_df)
    model_trainer.train_model()

    # Get model metrics
    model = model_trainer.aml.leader
    acc = model.accuracy()
    if isinstance(acc, list):
        acc = acc[0][1]

    # Prepare metrics and parameters
    metrics = {
        "logloss": model.logloss(),
        "auc": model.auc(),
        "accuracy": acc
    }

    params = {
        "max_models": model_trainer.max_models,
        "seed": model_trainer.seed,
        "max_runtime_secs": model_trainer.max_runtime_secs
    }

    # Get requirements path
    requirements_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "requirements.txt")

    # Get lists of functions used in feature engineering
    preprocessing_functions, engineering_functions = feature_engineer.get_used_functions()

    # Record experiment
    log.info("\nRecording experiment in MLflow...")
    experiment_recorder.record_experiment(
        experiment_name="Titanic4",
        model=model,
        metrics=metrics,
        params=params,
        requirements_path=requirements_path,
        training_data=train_df,
        preprocessing_functions=preprocessing_functions,
        engineering_functions=engineering_functions
    )

    # List all experiments
    log.info("\nListing all MLflow experiments...")
    experiment_recorder.list_experiments()

    # Make predictions and create submission
    predictions = model_trainer.make_predictions()
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Transported': predictions['predict'].as_data_frame().values.flatten()
    })
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    log.info("\nSubmission file created: submission.csv")

if __name__ == "__main__":
    main() 