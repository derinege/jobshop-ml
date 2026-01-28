"""
Data loading module for Excel file reading.

This module provides functionality to load Excel files containing scheduling
data. It focuses solely on file I/O and does not perform any preprocessing
or instance creation.
"""

import pandas as pd
from typing import Optional, Tuple
import config


class DataLoader:
    """
    Loads Excel files containing scheduling data.
    
    This class is responsible for reading Excel files and extracting raw data.
    It does not perform any data transformation or instance creation - those
    responsibilities are handled by the preprocessing module.
    
    Attributes:
    -----------
    islem_tam_path : str
        Path to the islem_tam_tablo.xlsx file.
    bold_sure_path : str
        Path to the bold_islem_sure_tablosu.xlsx file.
    df_tam : Optional[pd.DataFrame]
        Loaded data from islem_tam_tablo.xlsx.
    df_sure : Optional[pd.DataFrame]
        Loaded data from bold_islem_sure_tablosu.xlsx.
    bold_jobs : Optional[list]
        List of BOLD job identifiers extracted from df_tam.
    """
    
    def __init__(self, 
                 islem_tam_path: str = config.DATA_PATH_ISLEM_TAM,
                 bold_sure_path: str = config.DATA_PATH_BOLD_SURE):
        """
        Initialize data loader with file paths.
        
        Parameters:
        -----------
        islem_tam_path : str, optional
            Path to islem_tam_tablo.xlsx file. Defaults to config value.
        bold_sure_path : str, optional
            Path to bold_islem_sure_tablosu.xlsx file. Defaults to config value.
        """
        self.islem_tam_path = islem_tam_path
        self.bold_sure_path = bold_sure_path
        self.df_tam = None
        self.df_sure = None
        self.bold_jobs = None
    
    def load_data(self) -> 'DataLoader':
        """
        Load Excel files and extract BOLD job identifiers.
        
        This method reads both Excel files and filters for BOLD products.
        The data is stored in instance attributes for later use by preprocessing.
        
        Returns:
        --------
        DataLoader
            Returns self for method chaining.
        
        Raises:
        -------
        FileNotFoundError
            If either Excel file cannot be found.
        ValueError
            If required columns are missing from the Excel files.
        
        Notes:
        ------
        The method expects:
        - islem_tam_tablo.xlsx to have 'BOLD_FLAG' and 'TITLE' columns
        - bold_islem_sure_tablosu.xlsx to have standard operation columns
        """
        print(f"Loading {self.islem_tam_path}...")
        self.df_tam = pd.read_excel(self.islem_tam_path)
        
        # Validate required columns
        if 'BOLD_FLAG' not in self.df_tam.columns:
            raise ValueError("Column 'BOLD_FLAG' not found in islem_tam_tablo.xlsx")
        if 'TITLE' not in self.df_tam.columns:
            raise ValueError("Column 'TITLE' not found in islem_tam_tablo.xlsx")
        
        print(f"Loading {self.bold_sure_path}...")
        self.df_sure = pd.read_excel(self.bold_sure_path)
        
        # Filter for BOLD products
        bold_mask = self.df_tam['BOLD_FLAG'] == 1
        self.bold_jobs = self.df_tam[bold_mask]['TITLE'].unique().tolist()
        
        print(f"Found {len(self.bold_jobs)} BOLD jobs")
        print(f"Total operations in bold_islem_sure_tablosu: {len(self.df_sure)}")
        
        return self
    
    def get_bold_jobs(self) -> list:
        """
        Get list of BOLD job identifiers.
        
        Returns:
        --------
        list
            List of BOLD job TITLE values.
        
        Raises:
        -------
        RuntimeError
            If data has not been loaded yet.
        """
        if self.bold_jobs is None:
            raise RuntimeError("Data must be loaded first. Call load_data() before get_bold_jobs()")
        return self.bold_jobs.copy()
    
    def get_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get loaded dataframes.
        
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple of (df_tam, df_sure) dataframes.
        
        Raises:
        -------
        RuntimeError
            If data has not been loaded yet.
        """
        if self.df_tam is None or self.df_sure is None:
            raise RuntimeError("Data must be loaded first. Call load_data() before get_dataframes()")
        return self.df_tam.copy(), self.df_sure.copy()



