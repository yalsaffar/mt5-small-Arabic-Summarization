from transformers import AutoTokenizer
import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
import wandb
import os
from transformers import AutoModelForSeq2SeqLM

def load_tokenizer(model_name):
    """
    Load the tokenizer for a specific model.

    Args:
        model_name (str): The name of the model.

    Returns:
        tokenizer (AutoTokenizer): The loaded tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    Tokenize the input and label examples.

    Args:
        examples (dict): A dictionary containing the input and label examples.
        tokenizer (AutoTokenizer): The tokenizer to use.

    Returns:
        dict: A dictionary containing the tokenized input and label features.
    """
    input_feature = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=1024)
    label_feature = tokenizer(examples["summary"], padding="max_length", truncation=True, max_length=128)
    return {
        "input_ids": input_feature["input_ids"],
        "attention_mask": input_feature["attention_mask"],
        "labels": label_feature["input_ids"],
    }


def load_model(model_name, max_length=128, length_penalty=0.6, no_repeat_ngram_size=2, num_beams=15):
    """
    Load the model for sequence-to-sequence language modeling.

    Args:
        model_name (str): The name of the model.
        max_length (int): The maximum length of the generated sequences.
        length_penalty (float): The length penalty to apply during generation.
        no_repeat_ngram_size (int): The size of n-grams to avoid repeating during generation.
        num_beams (int): The number of beams to use during generation.

    Returns:
        device (torch.device): The device to use for training.
        model (AutoModelForSeq2SeqLM): The loaded model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mt5_config = AutoConfig.from_pretrained(
        model_name,
        max_length=max_length,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams,
    )
    model = (AutoModelForSeq2SeqLM
             .from_pretrained(model_name, config=mt5_config)
             .to(device))
    return device, model


def set_training_args(output_dir, log_level="error", num_train_epochs=10, learning_rate=5e-4, lr_scheduler_type="linear", warmup_steps=90, optim="adafactor", weight_decay=0.01, per_device_train_batch_size=2, per_device_eval_batch_size=1, gradient_accumulation_steps=16, evaluation_strategy="steps", eval_steps=100, predict_with_generate=True, generation_max_length=128, save_steps=500, logging_steps=10, push_to_hub=False):
    """
    Set the training arguments for the Seq2SeqTrainer.

    Args:
        output_dir (str): The output directory to save the trained model.
        log_level (str): The log level for training.
        num_train_epochs (int): The number of training epochs.
        learning_rate (float): The learning rate for training.
        lr_scheduler_type (str): The type of learning rate scheduler.
        warmup_steps (int): The number of warmup steps for the learning rate scheduler.
        optim (str): The optimizer to use for training.
        weight_decay (float): The weight decay for training.
        per_device_train_batch_size (int): The batch size per device for training.
        per_device_eval_batch_size (int): The batch size per device for evaluation.
        gradient_accumulation_steps (int): The number of gradient accumulation steps.
        evaluation_strategy (str): The evaluation strategy.
        eval_steps (int): The number of steps between evaluations.
        predict_with_generate (bool): Whether to use generation during evaluation.
        generation_max_length (int): The maximum length of generated sequences during evaluation.
        save_steps (int): The number of steps between model saves.
        logging_steps (int): The number of steps between logging.
        push_to_hub (bool): Whether to push the model to the Hugging Face model hub.

    Returns:
        training_args (Seq2SeqTrainingArguments): The training arguments.
    """
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        log_level=log_level,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        optim=optim,
        weight_decay=weight_decay,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy=evaluation_strategy,
        eval_steps=eval_steps,
        predict_with_generate=predict_with_generate,
        generation_max_length=generation_max_length,
        save_steps=save_steps,
        logging_steps=logging_steps,
        push_to_hub=push_to_hub
    )
    return training_args


def mt5_train(wandb_key, model, training_args, data_collator, metrics_func, dataset_dict, tokenizer):
    """
    Train the MT5 model using the Seq2SeqTrainer.

    Args:
        wandb_key (str): The WandB API key.
        model (AutoModelForSeq2SeqLM): The model to train.
        training_args (Seq2SeqTrainingArguments): The training arguments.
        data_collator (DataCollator): The data collator for training.
        metrics_func (Callable): The metrics function for evaluation.
        dataset_dict (dict): A dictionary containing the train and validation datasets.
        tokenizer (AutoTokenizer): The tokenizer to use.

    Returns:
        trainer (Seq2SeqTrainer): The trained trainer object.
    """
    wandb.login(key=wandb_key)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=metrics_func,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"].select(range(20)),
        tokenizer=tokenizer,
    )

    trainer.train()
    return trainer


def save_model(trainer, device, output_dir):
    """
    Save the fine-tuned model.

    Args:
        trainer (Seq2SeqTrainer): The trained trainer object.
        device (torch.device): The device used for training.
        output_dir (str): The output directory to save the model.

    Returns:
        model (AutoModelForSeq2SeqLM): The loaded model.
    """
    # Save fine-tuned model in local
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(trainer.model, "module"):
        trainer.model.module.save_pretrained(output_dir)
    else:
        trainer.model.save_pretrained(output_dir)

    # Load local model
    model = (AutoModelForSeq2SeqLM
             .from_pretrained(output_dir)
             .to(device))
    return model
