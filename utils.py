import pandas as pd
from torch.utils.data import Dataset


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




