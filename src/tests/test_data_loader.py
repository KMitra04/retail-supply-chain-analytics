import pandas as pd
import pytest
from src.data_loader import DataValidator, WalmartDataLoader

def test_check_missing_values_threshold():
    df = pd.DataFrame({
        'A': [1, None, 3, None],
        'B': [1, 2, 3, 4]
    })
    res = DataValidator.check_missing_values(df)
    assert 'A' in res
    assert pytest.approx(res['A'], rel=1e-3) == 50.0

def test_walmart_load_files_not_found():
    loader = WalmartDataLoader(data_dir='nonexistent_dir')
    with pytest.raises(FileNotFoundError):
        loader.load_all_files()
