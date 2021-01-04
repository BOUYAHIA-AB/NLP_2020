"""Peform hyperparemeters search"""

import argparse
import os
from subprocess import check_call
import sys

import utils


PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='experiments/all_parametres_science',
                    help='Directory containing params.json')
parser.add_argument('--data_dir', default='data/science', help="Directory containing the dataset")


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of the model with a set of hyperparameters in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights and log
        data_dir: (string) directory containing the dataset
        params: (dict) containing hyperparameters
    """
    # Create a new folder in parent_dir with unique_name "job_name"
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Write parameters in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    # Launch training with this config
    cmd = "{python} train.py --model_dir={model_dir} --data_dir {data_dir}".format(python=PYTHON, model_dir=model_dir,
                                                                                   data_dir=data_dir)
    print(cmd)
    check_call(cmd, shell=True)


if __name__ == "__main__":
    # Load the "reference" parameters from parent_dir json file
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Perform hypersearch over one parameter
    lstm_hidden_dims = [40, 50, 80, 100]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]
    embedding_dims = [40, 50, 80, 100]
    batch_sizes = [5, 10, 20, 50]

    for lstm_hidden_dim in lstm_hidden_dims:
        for learning_rate in learning_rates:
            for embedding_dim in embedding_dims:
                for batch_size in batch_sizes:
                    # Modify the relevant parameter in params
                    params.lstm_hidden_dim = lstm_hidden_dim
                    params.learning_rate = learning_rate
                    params.embedding_dim = embedding_dim
                    params.batch_size = batch_size

                    # Launch job (name has to be unique)
                    job_name = "lstm_hidden_dim_{}_{}_{}_{}".format(lstm_hidden_dim, learning_rate, embedding_dim, batch_size)
                    launch_training_job(args.parent_dir, args.data_dir, job_name, params)
