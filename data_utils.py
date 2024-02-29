import pandas as pd
from transformers import DataCollatorForSeq2Seq
import evaluate
import numpy as np
from nltk.tokenize import RegexpTokenizer
from datasets import Dataset, DatasetDict
from model_loader import tokenize_function

rouge_metric = evaluate.load("rouge")

class DataLoader():
    def __init__(self, path):
        """
        DataLoader class to load data from a CSV file.

        Args:
            path (str): The path to the CSV file.
        """
        self.path = path
       
   
    def load_dataframe(self):
        """
        Load the data from the CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: The loaded DataFrame.
        """
        self.df = pd.read_csv(self.path)
        return self.df

    def load_tokenized_dataframe(self, tokenizer):
        """
        Tokenize the data in the DataFrame using the provided tokenizer.

        Args:
            tokenizer: The tokenizer object to use for tokenization.

        Returns:
            pd.DataFrame: The tokenized DataFrame.
        """
        tokenized_data = self.df.apply(lambda x: tokenize_function(x, tokenizer), axis=1)
        self.tokenized_df = pd.DataFrame(list(tokenized_data))
        return self.tokenized_df


def data_collator(tokenizer, model):
    """
    Create a data collator for sequence-to-sequence models.

    Args:
        tokenizer: The tokenizer object to use for tokenization.
        model: The sequence-to-sequence model.

    Returns:
        DataCollatorForSeq2Seq: The data collator object.
    """
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        return_tensors="pt"
    )
    return data_collator


def tokenize_sentence(arg, tokenizer):
    """
    Tokenize a sentence using the provided tokenizer.

    Args:
        arg (str): The sentence to tokenize.
        tokenizer: The tokenizer object to use for tokenization.

    Returns:
        list: The tokenized sentence.
    """
    encoded_arg = tokenizer(arg)
    return tokenizer.convert_ids_to_tokens(encoded_arg.input_ids)


def metrics_func(eval_arg, tokenizer, rouge_metric):
    """
    Compute metrics for evaluation.

    Args:
        eval_arg (tuple): A tuple containing the predicted and actual labels.
        tokenizer: The tokenizer object to use for tokenization.
        rouge_metric: The metric object for computing ROUGE scores.

    Returns:
        float: The computed metric score.
    """
    preds, labels = eval_arg
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    text_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    text_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    text_preds = [(p if p.endswith(("!", "！", "?", "？", "。")) else p + "。") for p in text_preds]
    text_labels = [(l if l.endswith(("!", "！", "?", "？", "。")) else l + "。") for l in text_labels]
    sent_tokenizer_jp = RegexpTokenizer(u'[^!！?？。]*[!！?？。]')
    text_preds = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(p))) for p in text_preds]
    text_labels = ["\n".join(np.char.strip(sent_tokenizer_jp.tokenize(l))) for l in text_labels]
    return rouge_metric.compute(
        predictions=text_preds,
        references=text_labels,
        tokenizer=tokenize_sentence
    )


def split_dataframe(tokenized_df):
    """
    Split the tokenized DataFrame into train, test, and validation datasets.

    Args:
        tokenized_df: The tokenized DataFrame.

    Returns:
        DatasetDict: A dictionary containing the train, test, and validation datasets.
    """
    train_testvalid = tokenized_df.sample(frac=1, random_state=42)
    train_valid_split = int(0.8 * len(train_testvalid))
    train_df = train_testvalid[:train_valid_split]
    test_valid_df = train_testvalid[train_valid_split:]
    test_valid_split = int(0.5 * len(test_valid_df))
    test_df = test_valid_df[:test_valid_split]
    valid_df = test_valid_df[test_valid_split:]

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'validation': valid_dataset
    })

    return dataset_dict
