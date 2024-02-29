import argparse
from data_utils import *
from model_loader import *

def train_mt5(path_to_data, model_name, output_dir, wandb_key):
    """
    Trains an MT5 model for Arabic summarization.

    Args:
        path_to_data (str): The path to the data used for training.
        model_name (str): The name of the MT5 model to be used.
        output_dir (str): The directory where the trained model will be saved.
        wandb_key (str): The API key for Weights & Biases (wandb) integration.

    Returns:
        None
    """
    DataLoader_class = DataLoader(path_to_data)
    tokenizer = load_tokenizer(model_name)
    tokenized_df = DataLoader_class.load_tokenized_dataframe(tokenizer)
    device, model = load_model(model_name)
    data_collator = data_collator(tokenizer, model)
    dataset_dict = split_dataframe(tokenized_df)
    training_args = set_training_args(output_dir)
    model_trainer = mt5_train(wandb_key, model, training_args, data_collator, metrics_func, dataset_dict, tokenizer)
    save_model(model_trainer, device, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MT5 model")
    parser.add_argument("--path_to_data", type=str, help="Path to data")
    parser.add_argument("--model_name", type=str, help="Name of the model")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--wandb_key", type=str, help="Wandb API key")
    args = parser.parse_args()

    train_mt5(args.path_to_data, args.model_name, args.output_dir, args.wandb_key)
