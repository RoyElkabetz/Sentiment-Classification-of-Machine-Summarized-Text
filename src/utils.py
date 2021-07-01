import torch
import pandas as pd
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


class IMDBfromCSV_DataModule(pl.LightningDataModule):
    """IMDB Data module for pytorch-lightning"""
    def __init__(self, vocab, tokenizer, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.train_dataset = Dataset()
        self.valid_dataset = Dataset()
        self.test_dataset = Dataset()

    def setup(self, stage=None):
        self.train_dataset = DataFrameDataset(self.data_dir + '/train.csv', only_columns=['label', 'text'])
        self.valid_dataset = DataFrameDataset(self.data_dir + '/valid.csv', only_columns=['label', 'text'])
        self.test_dataset = DataFrameDataset(self.data_dir + '/test.csv', only_columns=['label', 'summary'])

    def text_pipeline(self, x):
        return self.vocab(self.tokenizer(x))

    def label_pipeline(self, x):
        num = 0
        if x == 'pos':
            num = 1
        return num

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, collate_fn=self.collate_batch)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_batch)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, collate_fn=self.collate_batch)

    def collate_batch(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.label_pipeline(_label))
            processed_text = torch.tensor(self.text_pipeline(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list, text_list, offsets


class IMDBClassificationTask(pl.LightningModule):
    """IMDB task module for pytorch-lightning"""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, text, offsets):
        return self.model(text=text, offsets=offsets)

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {'train_acc': acc, 'train_loss': loss}
        self.log_dict(metrics, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {'valid_acc': acc, 'valid_loss': loss}
        self.log_dict(metrics, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {'test_acc': acc, 'test_loss': loss}
        self.log_dict(metrics)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        (label, text, offsets) = batch
        pred_label = self.model(text, offsets)
        loss = F.cross_entropy(pred_label, label)
        acc = (pred_label.argmax(1) == label).sum().item()
        count = len(label)
        acc /= count
        return loss, acc

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=5.)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
        lr_dict = {'optimizer': optimizer,
                   'lr_scheduler': {'scheduler': scheduler,
                                    'interval': 'epoch',
                                    'monitor': 'valid_acc',
                                    }
                   }
        return lr_dict


class NewsSummaryDataset(Dataset):

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

        # i added self
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


