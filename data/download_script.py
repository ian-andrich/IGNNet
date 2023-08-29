import openml
import logging


# SUITE_ID = 336 # Regression on numerical features
SUITE_ID = 337  # Classification on numerical features
# SUITE_ID = 335 # Regression on numerical and categorical features
SUITE_ID = 334  # Classification on numerical and categorical features
benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
if benchmark_suite.tasks is not None:
    for task_id in benchmark_suite.tasks:  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        # dataset = task.get_dataset()
        """
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        """


msg = "Data downloaded by default to ~/.openml unless configured otherwise"
logging.getLogger(__name__).warning(msg)
