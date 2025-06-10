import pytest
import pandas as pd
import numpy as np
from data_processing import process_bathroom_count

class TestProcessBathroomCount:
    
    def test_colon_format_basic(self):
        """Test basic colon format: '2:1' -> (2 full, 1 half)"""
        result = process_bathroom_count("2:1")
        assert result == (2, 1)
    
    def test_f_p_format_no_spaces(self):
        """Test F/P format without spaces: '2F1P' -> (2 full, 1 half)"""
        result = process_bathroom_count("2F1P")
        assert result == (2, 1)
    
    def test_f_p_format_with_spaces(self):
        """Test F/P format with spaces: '2F 1P' -> (2 full, 1 half)"""
        result = process_bathroom_count("2F 1P")
        assert result == (2, 1)
    
    def test_f_h_format_mixed_case(self):
        """Test F/H format with mixed case: '3f 2H' -> (3 full, 2 half)"""
        result = process_bathroom_count("3f 2H")
        assert result == (3, 2)
    
    def test_f_p_format_complex(self):
        """Test complex F/P format with extra characters: '4F+1P' -> (4 full, 1 half)"""
        result = process_bathroom_count("4F+1P")
        assert result == (4, 1)
    
    def test_single_number_fallback(self):
        """Test single number fallback: '3' -> (3 full, 0 half)"""
        result = process_bathroom_count("3")
        assert result == (3, 0)
    
    def test_zero_values(self):
        """Test edge case with zero values: '0F1P' -> (0 full, 1 half)"""
        result = process_bathroom_count("0F1P")
        assert result == (0, 1)

# P2P Tests (should pass in incomplete version)
class TestProcessBathroomCountP2P:

    def test_null_input(self):
        """Test null input handling"""
        result = process_bathroom_count(None)
        assert pd.isna(result[0]) and pd.isna(result[1])
    
    def test_nan_input(self):
        """Test NaN input handling"""
        result = process_bathroom_count(np.nan)
        assert pd.isna(result[0]) and pd.isna(result[1])
    
    def test_invalid_string(self):
        """Test invalid string handling"""
        result = process_bathroom_count("invalid")
        assert pd.isna(result[0]) and pd.isna(result[1])
    
    def test_function_returns_tuple(self):
        """P2P: Function should always return a tuple of length 2"""
        test_cases = ["2:1", "2F1P", "3", None, "invalid", np.nan]
        for case in test_cases:
            result = process_bathroom_count(case)
            assert isinstance(result, tuple)
            assert len(result) == 2
    
    def test_function_handles_none_gracefully(self):
        """P2P: Function should not crash on None input"""
        try:
            result = process_bathroom_count(None)
            # Should not raise an exception
            assert True
        except Exception:
            assert False, "Function should handle None input gracefully"

if __name__ == '__main__':
    pytest.main([__file__])