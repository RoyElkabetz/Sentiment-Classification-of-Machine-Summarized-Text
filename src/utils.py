import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast as T5Tokenizer
)



class DataFrameDataset(Dataset):
    """This class create a torch.utils.data.Dataset from a pandas.DataFrame or from a CSV file."""

    def __init__(self, csv_file_path=None, pd_dataframe=None, only_columns=None):
        """
          Args:
          csv_file_path (string): Path to the csv file with annotations.
          pd_dataframe (Pandas DataFrame): A Pandas DataFrame with containing the data.
          only_columns (list): A List of only column names from the data you want to use.
        """
        if isinstance(pd_dataframe, pd.DataFrame):
            self.df = pd_dataframe
        else:
            self.df = pd.read_csv(csv_file_path)

        if only_columns is not None:
            if isinstance(only_columns, list):
                for item in only_columns:
                    if item not in self.df.columns:
                        raise ValueError(f"Got a column name '{item}' in only_columns which is not in data columns.")
                    self.only_columns = only_columns
            else:
                raise TypeError(f"only_columns must be a <class 'list'>, instead got a {type(only_columns)}.")
        else:
            self.only_columns = list(self.df.columns)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx][self.only_columns]
        row_list = [item for item in row]
        return row_list


class DataLogger:
    """Simple data logger class"""
    def __init__(self, logged_arguments: list):
        """
          Args:
          logged_arguments (list): a list of strings - names of variables that we would like
          to save along the training process.
        """
        self.logger = {}

        for item in logged_arguments:
            self.logger[item] = []

    def log_up(self, variable, value):
        """append a measurement to variable"""
        self.logger[variable].append(value)

    def dict_log(self, dict_bag):
        """append a dictionary of measurements to variables"""
        for item in dict_bag:
            self.logger[item].append(dict_bag[item])

    def save_log(self, save_path):
        """save the logger as json file"""
        json.dump(self.logger, open(save_path + ".json", 'w'))
        print('Logger was saved.')

    def load_log(self, load_path):
        """load a json file as a logger"""
        self.logger = json.load(open(load_path + ".json"))
        print('Logger was loaded.')



class NewsSummaryDataset(Dataset):
    """Dataset class for the News Summary dataset"""
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: T5Tokenizer,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128
    ):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]

        text = data_row['text']
        summary = data_row['summary']

        text_encoding = self.tokenizer(
            text,
            max_length=self.text_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        summary_encoding = self.tokenizer(
            summary,
            max_length=self.summary_max_token_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100

        return dict(
            text=text,
            summary=summary,
            text_input_ids=text_encoding['input_ids'].flatten(),
            text_attention_mask=text_encoding['attention_mask'].flatten(),
            labels=labels.flatten(),
            labels_attention_mask=summary_encoding['attention_mask'].flatten()
        )


class NewsSummaryDataModule(pl.LightningDataModule):
    """A data module class with pytorch lightning for the training of T5"""
    def __init__(
        self,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        tokenizer: T5Tokenizer,
        batch_size: int = 10,
        text_max_token_len: int = 512,
        summary_max_token_len: int = 128
    ):
        super().__init__()

        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len

    def setup(self, stage=None):
        self.train_dataset = NewsSummaryDataset(
            self.train_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

        self.valid_dataset = NewsSummaryDataset(
            self.valid_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

        self.test_dataset = NewsSummaryDataset(
            self.test_df,
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )


