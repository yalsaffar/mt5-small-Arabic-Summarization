# Arabic News Article Summarization with MT5

This project fine-tunes the `google/mt5-small` model on the BBC Arabic news dataset for the task of summarization. It involves summarizing news articles into concise summaries, utilizing a Transformer-based model for its state-of-the-art performance in natural language understanding and generation tasks.


## Introduction

Leveraging the power of the `google/mt5-small` model, this project aims at harnessing its multilingual processing capabilities to address the unique linguistic nuances of Arabic. The project utilizes the Transformers library for fine-tuning the model on the BBC Arabic news dataset, focusing on summarization tasks. Through custom training configurations and optimizations, the model is enhanced to generate accurate and concise summaries of Arabic news articles. The Seq2Seq training framework supports an efficient training loop, and by employing evaluation metrics such as ROUGE scores, the project ensures the summaries' quality closely matches that of human experts.


## Installation

To set up this project for use or development, clone the repository and install the required packages:

```bash
git clone https://github.com/yalsaffar/mt5-small-Arabic-Summarization.git
cd mt5-small-Arabic-Summarization
pip install -r requirements.txt
```

## Hugging Face  Implementation
The fine-tuned model is available on Hugging Face, providing an easy way to integrate and use the summarization model in various applications. The model can be accessed and utilized directly through the Hugging Face platform.

- [Hugging Face for hosting the fine-tuned model](https://huggingface.co/yalsaffar/mt5-small-Arabic-Summarization)
## Usage

There are two ways to replicate the model fine-tuning:

1. **Using the Command Line:**

```bash
python train_mt5.py --path_to_data [path_to_data] --model_name google/mt5-small --output_dir [output_directory] --wandb_key [your_wandb_key]
```

2. **Using the Jupyter Notebook:**
Open the `training_mt5.ipynb` notebook and follow the instructions within to fine-tune the model interactively.

## Dataset

The dataset used for training is the BBC Arabic news dataset, comprising news articles and their corresponding summaries. The dataset is split into 32,000 training rows, 4,000 testing, and 4,000 validation rows.

- Dataset source: [BBC Arabic News Data](https://www.kaggle.com/datasets/fadyelkbeer/arabic-summarization-bbc-news)

## Model

`google/mt5-small` represents a significant advancement in the domain of multilingual natural language processing. Originating from the T5 (Text-to-Text Transfer Transformer) family, mT5 extends the capabilities of T5 to a global scale by being pre-trained on a diverse dataset covering 101 languages, including Arabic​​​​. 

- [Original MT5 model by Google](https://huggingface.co/google/mt5-small)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




