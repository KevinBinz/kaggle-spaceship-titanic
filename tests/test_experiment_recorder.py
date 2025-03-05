import os
import sys
import unittest
import shutil
import pandas as pd
import numpy as np
import mlflow
import requests
import time
import platform
import tempfile

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
if src_dir not in sys.path:
    sys.path.append(src_dir)

from experiment_recorder import ExperimentRecorder
from logging_utils import log

class MockModel:
    """A mock model class that mimics H2O model interface."""
    def __init__(self):
        self.accuracy = lambda: 0.85
        self.auc = lambda: 0.82
        self.logloss = lambda: 0.45

class TestExperimentRecorder(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        log.info("\nSetting up test...")
        # Create a simple test model
        self.test_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100, dtype=np.int64)  # Explicitly set dtype to int64
        })
        
        # Create a mock model
        self.test_model = MockModel()
        
        # Create test metrics and parameters
        self.test_metrics = {
            'accuracy': 0.85,
            'auc': 0.82,
            'logloss': 0.45
        }
        self.test_params = {
            'max_models': '1',
            'seed': '42',
            'max_runtime_secs': '30'
        }
        
        # Create test preprocessing and engineering functions
        self.test_preprocessing_functions = ['_extract_cabin_components', '_extract_name_components']
        self.test_engineering_functions = ['create_cabin_features', 'create_age_features']
        
        # Create requirements.txt for testing
        self.requirements_path = os.path.join(current_dir, 'requirements.txt')
        with open(self.requirements_path, 'w') as f:
            f.write('mlflow==2.8.1\npandas==2.0.0\nnumpy==1.24.0\n')

    def tearDown(self):
        """Clean up test fixtures after each test method."""
        log.info("\nCleaning up...")
        if os.path.exists(self.requirements_path):
            os.remove(self.requirements_path)

    def test_get_experiments(self):
        """Test getting experiment details."""
        log.info("\nTesting get_experiments...")
        recorder = ExperimentRecorder(tracking_uri="http://localhost:5000")
        
        # Record a test experiment first
        recorder.record_experiment(
            experiment_name="test_experiment",
            model=self.test_model,
            metrics=self.test_metrics,
            params=self.test_params,
            requirements_path=self.requirements_path,
            description="Test experiment for get_experiments"
        )
        
        # Get experiments
        experiments = recorder.get_experiments()
        
        # Verify experiments is a list
        self.assertIsInstance(experiments, list)
        self.assertGreater(len(experiments), 0)
        
        # Find our test experiment
        test_exp = next((exp for exp in experiments if exp['experiment_name'] == 'test_experiment'), None)
        self.assertIsNotNone(test_exp)
        
        # Verify experiment details
        self.assertEqual(test_exp['experiment_name'], 'test_experiment')
        self.assertIsNotNone(test_exp['experiment_id'])
        self.assertEqual(test_exp['accuracy'], 0.85)
        self.assertIsNotNone(test_exp['run_id'])
        self.assertEqual(test_exp['description'], 'Test experiment for get_experiments')

    def test_display_experiments(self):
        """Test displaying experiment details."""
        log.info("\nTesting display_experiments...")
        recorder = ExperimentRecorder(tracking_uri="http://localhost:5000")
        
        # Record multiple test experiments with different accuracies
        recorder.record_experiment(
            experiment_name="test_experiment_1",
            model=self.test_model,
            metrics={'accuracy': 0.85},
            params=self.test_params,
            requirements_path=self.requirements_path,
            description="First test experiment"
        )
        
        recorder.record_experiment(
            experiment_name="test_experiment_2",
            model=self.test_model,
            metrics={'accuracy': 0.82},
            params=self.test_params,
            requirements_path=self.requirements_path,
            description="Second test experiment"
        )
        
        # Get the experiment summary DataFrame
        summary_df = recorder.display_experiments()
        
        # Verify DataFrame structure
        self.assertIsInstance(summary_df, pd.DataFrame)
        self.assertEqual(len(summary_df.columns), 4)
        self.assertEqual(list(summary_df.columns), ['Experiment Name', 'Run ID', 'Accuracy', 'Description'])
        
        # Verify sorting by accuracy
        self.assertTrue(summary_df['Accuracy'].is_monotonic_decreasing)
        
        # Verify experiment details
        exp1_row = summary_df[summary_df['Experiment Name'] == 'test_experiment_1'].iloc[0]
        self.assertEqual(exp1_row['Accuracy'], 0.85)
        self.assertEqual(exp1_row['Description'], 'First test experiment')
        
        exp2_row = summary_df[summary_df['Experiment Name'] == 'test_experiment_2'].iloc[0]
        self.assertEqual(exp2_row['Accuracy'], 0.82)
        self.assertEqual(exp2_row['Description'], 'Second test experiment')
        
        # Verify handling of None descriptions
        recorder.record_experiment(
            experiment_name="test_experiment_3",
            model=self.test_model,
            metrics={'accuracy': 0.80},
            params=self.test_params,
            requirements_path=self.requirements_path
        )
        
        summary_df = recorder.display_experiments()
        exp3_row = summary_df[summary_df['Experiment Name'] == 'test_experiment_3'].iloc[0]
        self.assertEqual(exp3_row['Description'], 'No description')

    def test_http_server_connection(self):
        """Test that we can connect to the HTTP server and perform basic operations."""
        log.info("\nTesting HTTP server connection...")
        recorder = ExperimentRecorder(tracking_uri="http://localhost:5000")
        
        # Test connection
        log.info("Testing HTTP connection...")
        self.assertTrue(recorder._is_mlflow_server_running())
        
        # Test listing experiments
        log.info("Testing experiment listing...")
        mlflow.set_tracking_uri(recorder.tracking_uri)
        experiments = mlflow.search_experiments()
        self.assertIsNotNone(experiments)
        self.assertGreater(len(experiments), 0)

    def test_basic_experiment_recording(self):
        """Test basic experiment recording functionality."""
        log.info("\nTesting basic experiment recording...")
        recorder = ExperimentRecorder(tracking_uri="http://localhost:5000")
        
        # Record experiment with training data
        recorder.record_experiment(
            experiment_name="test_experiment",
            model=self.test_model,
            metrics=self.test_metrics,
            params=self.test_params,
            requirements_path=self.requirements_path,
            training_data=self.test_data
        )
        
        # Verify experiment was created
        mlflow.set_tracking_uri("http://localhost:5000")
        experiments = mlflow.search_experiments()
        experiment_names = [exp.name for exp in experiments]
        self.assertIn("test_experiment", experiment_names)
        
        # Get the experiment ID
        experiment = mlflow.get_experiment_by_name("test_experiment")
        self.assertIsNotNone(experiment)
        
        # Verify run was created
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        self.assertGreater(len(runs), 0)
        
        # Verify parameters were logged
        latest_run = runs.iloc[0]
        for key, value in self.test_params.items():
            self.assertEqual(str(value), latest_run[f'params.{key}'])
        
        # Verify metrics were logged
        for key, value in self.test_metrics.items():
            self.assertAlmostEqual(value, latest_run[f'metrics.{key}'])
            
        # Verify artifact was created by checking artifact URI
        artifact_uri = latest_run['artifact_uri']
        self.assertIn('mlflow-artifacts:', artifact_uri)
        
        # Verify DataFrame headers were logged as tag
        self.assertIn('tags.dataframe_headers', latest_run)
        logged_headers = set(latest_run['tags.dataframe_headers'].split(','))
        expected_headers = set(self.test_data.columns)
        self.assertEqual(logged_headers, expected_headers)
        
        # Download and verify the DataFrame
        client = mlflow.tracking.MlflowClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            client.download_artifacts(latest_run['run_id'], 'data', temp_dir)
            # The artifact is downloaded as a directory named 'data'
            data_dir = os.path.join(temp_dir, 'data')
            self.assertTrue(os.path.isdir(data_dir))
            # List files in the data directory
            downloaded_files = os.listdir(data_dir)
            self.assertEqual(len(downloaded_files), 1)
            # Read the first (and only) file
            loaded_df = pd.read_csv(os.path.join(data_dir, downloaded_files[0]))
            log.info(f"Loaded DataFrame shape: {loaded_df.shape}")
            log.info(f"Test DataFrame shape: {self.test_data.shape}")
            pd.testing.assert_frame_equal(self.test_data, loaded_df)

    def test_function_name_logging(self):
        """Test logging of preprocessing and engineering function names."""
        log.info("\nTesting function name logging...")
        recorder = ExperimentRecorder(tracking_uri="http://localhost:5000")
        
        # Record experiment with function names
        recorder.record_experiment(
            experiment_name="test_experiment",
            model=self.test_model,
            metrics=self.test_metrics,
            params=self.test_params,
            requirements_path=self.requirements_path,
            training_data=self.test_data,
            preprocessing_functions=self.test_preprocessing_functions,
            engineering_functions=self.test_engineering_functions
        )
        
        # Verify experiment was created
        mlflow.set_tracking_uri("http://localhost:5000")
        experiment = mlflow.get_experiment_by_name("test_experiment")
        self.assertIsNotNone(experiment)
        
        # Get the latest run
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        latest_run = runs.iloc[0]
        
        # Verify preprocessing functions were logged
        self.assertIn('tags.preprocessing_functions', latest_run)
        logged_preprocessing = set(latest_run['tags.preprocessing_functions'].split(','))
        self.assertEqual(logged_preprocessing, set(self.test_preprocessing_functions))
        
        # Verify engineering functions were logged
        self.assertIn('tags.engineering_functions', latest_run)
        logged_engineering = set(latest_run['tags.engineering_functions'].split(','))
        self.assertEqual(logged_engineering, set(self.test_engineering_functions))

    def test_server_connection_error(self):
        """Test error handling when MLflow server is not running."""
        log.info("\nTesting server connection error handling...")
        recorder = ExperimentRecorder(tracking_uri="http://nonexistent-server:5000")
        
        # Attempt to record experiment should raise RuntimeError
        with self.assertRaises(RuntimeError) as context:
            recorder.record_experiment(
                experiment_name="test_experiment",
                model=self.test_model,
                metrics=self.test_metrics,
                params=self.test_params,
                requirements_path=self.requirements_path
            )
        
        self.assertIn("MLflow server not running", str(context.exception))

    def test_invalid_tracking_uri(self):
        """Test error handling for invalid tracking URI."""
        log.info("\nTesting invalid tracking URI handling...")
        with self.assertRaises(ValueError) as context:
            ExperimentRecorder(tracking_uri="invalid://uri")
        
        self.assertIn("tracking_uri must be an HTTP(S) URL", str(context.exception))

    def test_experiment_description(self):
        """Test logging of experiment description."""
        log.info("\nTesting experiment description logging...")
        recorder = ExperimentRecorder(tracking_uri="http://localhost:5000")
        
        # Test description
        test_description = "This is a test experiment with a detailed description"
        
        # Record experiment with description
        recorder.record_experiment(
            experiment_name="test_experiment",
            model=self.test_model,
            metrics=self.test_metrics,
            params=self.test_params,
            requirements_path=self.requirements_path,
            description=test_description
        )
        
        # Verify experiment was created
        mlflow.set_tracking_uri("http://localhost:5000")
        experiment = mlflow.get_experiment_by_name("test_experiment")
        self.assertIsNotNone(experiment)
        
        # Get the latest run
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        latest_run = runs.iloc[0]
        
        # Verify description was logged
        self.assertIn('tags.description', latest_run)
        self.assertEqual(latest_run['tags.description'], test_description)
        
        # Test without description
        recorder.record_experiment(
            experiment_name="test_experiment",
            model=self.test_model,
            metrics=self.test_metrics,
            params=self.test_params,
            requirements_path=self.requirements_path
        )
        
        # Get the latest run
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        latest_run = runs.iloc[0]
        
        # Verify description tag has value None
        self.assertIn('tags.description', latest_run)
        self.assertIsNone(latest_run['tags.description'])

    def test_dataframe_artifact_logging(self):
        """Test logging of DataFrame as artifact."""
        log.info("\nTesting DataFrame artifact logging...")
        recorder = ExperimentRecorder(tracking_uri="http://localhost:5000")
        
        # Create a small test DataFrame
        test_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': ['a', 'b', 'c'],
            'target': [0, 1, 0]
        })
        
        # Record experiment with DataFrame
        recorder.record_experiment(
            experiment_name="test_experiment",
            model=self.test_model,
            metrics=self.test_metrics,
            params=self.test_params,
            requirements_path=self.requirements_path,
            training_data=test_df
        )
        
        # Verify experiment was created
        mlflow.set_tracking_uri("http://localhost:5000")
        experiment = mlflow.get_experiment_by_name("test_experiment")
        self.assertIsNotNone(experiment)
        
        # Get the latest run
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        latest_run = runs.iloc[0]
        
        # Verify artifact was created by checking artifact URI
        artifact_uri = latest_run['artifact_uri']
        self.assertIn('mlflow-artifacts:', artifact_uri)
        
        # Verify DataFrame headers were logged as tag
        self.assertIn('tags.dataframe_headers', latest_run)
        logged_headers = set(latest_run['tags.dataframe_headers'].split(','))
        expected_headers = set(test_df.columns)
        self.assertEqual(logged_headers, expected_headers)
        
        # Download and verify the DataFrame
        client = mlflow.tracking.MlflowClient()
        with tempfile.TemporaryDirectory() as temp_dir:
            client.download_artifacts(latest_run['run_id'], 'data', temp_dir)
            # The artifact is downloaded as a directory named 'data'
            data_dir = os.path.join(temp_dir, 'data')
            self.assertTrue(os.path.isdir(data_dir))
            # List files in the data directory
            downloaded_files = os.listdir(data_dir)
            self.assertEqual(len(downloaded_files), 1)
            # Read the first (and only) file
            loaded_df = pd.read_csv(os.path.join(data_dir, downloaded_files[0]))
            pd.testing.assert_frame_equal(test_df, loaded_df)

    def test_exclude_test_experiments(self):
        """Test excluding test experiments from get_experiments."""
        log.info("\nTesting exclude_test_experiments...")
        recorder = ExperimentRecorder(tracking_uri="http://localhost:5000")
        
        # Record multiple experiments including test experiments
        recorder.record_experiment(
            experiment_name="test_experiment_1",
            model=self.test_model,
            metrics={'accuracy': 0.85},
            params=self.test_params,
            requirements_path=self.requirements_path,
            description="First test experiment"
        )
        
        recorder.record_experiment(
            experiment_name="production_experiment",
            model=self.test_model,
            metrics={'accuracy': 0.82},
            params=self.test_params,
            requirements_path=self.requirements_path,
            description="Production experiment"
        )
        
        # Get all experiments (including test experiments)
        all_experiments = recorder.get_experiments()
        self.assertGreater(len(all_experiments), 0)
        
        # Verify test experiments are included
        test_experiments = [exp for exp in all_experiments if 'test' in exp['experiment_name'].lower()]
        self.assertGreater(len(test_experiments), 0)
        
        # Get experiments excluding test experiments
        filtered_experiments = recorder.get_experiments(exclude_test_experiment=True)
        
        # Verify test experiments are excluded
        test_experiments = [exp for exp in filtered_experiments if 'test' in exp['experiment_name'].lower()]
        self.assertEqual(len(test_experiments), 0)
        
        # Verify production experiment is still included
        production_experiments = [exp for exp in filtered_experiments if exp['experiment_name'] == 'production_experiment']
        self.assertEqual(len(production_experiments), 1)

if __name__ == '__main__':
    unittest.main() 