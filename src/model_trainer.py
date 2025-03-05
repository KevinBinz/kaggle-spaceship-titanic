import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import os

class ModelTrainer:
    def __init__(self, max_models=10, max_runtime_secs=7200, seed=42):
        """Initialize the ModelTrainer with H2O AutoML parameters."""
        self.max_models = max_models
        self.max_runtime_secs = max_runtime_secs
        self.seed = seed
        self.aml = None
        self.train_frame = None
        self.test_frame = None

    def init_h2o(self):
        """Initialize H2O cluster."""
        h2o.init()

    def load_data(self, train_df, test_df):
        """Load pandas DataFrames into H2O frames."""
        self.train_frame = h2o.H2OFrame(train_df)
        self.test_frame = h2o.H2OFrame(test_df)
        
        if 'Transported' in self.train_frame.columns:
            self.train_frame['Transported'] = self.train_frame['Transported'].asfactor()

    def train_model(self, target='Transported', exclude_columns=None):
        """Train H2O AutoML model."""
        if exclude_columns is None:
            exclude_columns = []

        x = [col for col in self.train_frame.columns if col not in [target] + exclude_columns]
        y = target

        self.aml = H2OAutoML(
            max_models=self.max_models,
            seed=self.seed,
            max_runtime_secs=self.max_runtime_secs,
            sort_metric='accuracy'
        )
        
        self.aml.train(x=x, y=y, training_frame=self.train_frame)
        return self.aml

    def get_leaderboard(self):
        """Get the leaderboard of trained models."""
        if self.aml is None:
            raise ValueError("No models have been trained yet.")
        return self.aml.leaderboard

    def get_feature_importance(self, use_pandas=True):
        """Get feature importance from the best model."""
        if self.aml is None:
            raise ValueError("No models have been trained yet.")
        return self.aml.leader.varimp(use_pandas=use_pandas)

    def plot_feature_importance(self):
        """Plot feature importance from the best model."""
        if self.aml is None:
            raise ValueError("No models have been trained yet.")
        return self.aml.leader.varimp_plot()

    def make_predictions(self, new_data=None):
        """Make predictions using the best model."""
        if self.aml is None:
            raise ValueError("No models have been trained yet.")
        
        if new_data is None:
            new_data = self.test_frame
        elif isinstance(new_data, pd.DataFrame):
            new_data = h2o.H2OFrame(new_data)
            
        return self.aml.leader.predict(new_data) 