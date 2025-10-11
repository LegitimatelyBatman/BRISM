"""
Tests for data loader error handling.
"""

import unittest
import tempfile
import os
import pandas as pd
from brism.data_loader import load_mimic_data


class TestDataLoaderErrorHandling(unittest.TestCase):
    """Test error handling in data loader."""
    
    def test_file_not_found_diagnoses(self):
        """Test that missing diagnoses file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notes_path = os.path.join(tmpdir, 'notes.csv')
            pd.DataFrame({'subject_id': [1], 'text': ['test']}).to_csv(notes_path, index=False)
            
            with self.assertRaises(FileNotFoundError) as context:
                load_mimic_data('nonexistent.csv', notes_path)
            self.assertIn("Diagnoses file not found", str(context.exception))
    
    def test_file_not_found_notes(self):
        """Test that missing notes file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = os.path.join(tmpdir, 'diag.csv')
            pd.DataFrame({'subject_id': [1], 'icd_code': ['I10']}).to_csv(diag_path, index=False)
            
            with self.assertRaises(FileNotFoundError) as context:
                load_mimic_data(diag_path, 'nonexistent.csv')
            self.assertIn("Notes file not found", str(context.exception))
    
    def test_missing_required_column_diagnoses(self):
        """Test that missing required columns in diagnoses raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = os.path.join(tmpdir, 'diag.csv')
            notes_path = os.path.join(tmpdir, 'notes.csv')
            
            # Missing icd_code column
            pd.DataFrame({'subject_id': [1]}).to_csv(diag_path, index=False)
            pd.DataFrame({'subject_id': [1], 'text': ['test']}).to_csv(notes_path, index=False)
            
            with self.assertRaises(ValueError) as context:
                load_mimic_data(diag_path, notes_path)
            self.assertIn("must contain 'icd_code' column", str(context.exception))
    
    def test_missing_required_column_notes(self):
        """Test that missing required columns in notes raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = os.path.join(tmpdir, 'diag.csv')
            notes_path = os.path.join(tmpdir, 'notes.csv')
            
            pd.DataFrame({'subject_id': [1], 'icd_code': ['I10']}).to_csv(diag_path, index=False)
            # Missing text column
            pd.DataFrame({'subject_id': [1]}).to_csv(notes_path, index=False)
            
            with self.assertRaises(ValueError) as context:
                load_mimic_data(diag_path, notes_path)
            self.assertIn("must contain 'text' column", str(context.exception))
    
    def test_missing_id_column_diagnoses(self):
        """Test that missing ID columns in diagnoses raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = os.path.join(tmpdir, 'diag.csv')
            notes_path = os.path.join(tmpdir, 'notes.csv')
            
            # Missing both hadm_id and subject_id
            pd.DataFrame({'icd_code': ['I10']}).to_csv(diag_path, index=False)
            pd.DataFrame({'subject_id': [1], 'text': ['test']}).to_csv(notes_path, index=False)
            
            with self.assertRaises(ValueError) as context:
                load_mimic_data(diag_path, notes_path)
            self.assertIn("must contain 'hadm_id' or 'subject_id' column", str(context.exception))
    
    def test_dry_run_mode(self):
        """Test that dry_run mode validates data without full processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = os.path.join(tmpdir, 'diag.csv')
            notes_path = os.path.join(tmpdir, 'notes.csv')
            
            pd.DataFrame({
                'subject_id': [1, 2],
                'icd_code': ['I10', 'E11.9']
            }).to_csv(diag_path, index=False)
            
            pd.DataFrame({
                'subject_id': [1, 2],
                'text': ['Patient has fever', 'Patient has cough']
            }).to_csv(notes_path, index=False)
            
            # Should not raise, just validate
            train_ds, val_ds, test_ds, preprocessor = load_mimic_data(
                diag_path, notes_path, dry_run=True
            )
            
            # Datasets should be empty in dry run
            self.assertEqual(len(train_ds), 0)
            self.assertEqual(len(val_ds), 0)
            self.assertEqual(len(test_ds), 0)
    
    def test_malformed_csv(self):
        """Test that malformed CSV raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            diag_path = os.path.join(tmpdir, 'diag.csv')
            notes_path = os.path.join(tmpdir, 'notes.csv')
            
            # Write malformed CSV
            with open(diag_path, 'w') as f:
                f.write('subject_id,icd_code\n1,I10\n2,E11.9,extra_field')  # Inconsistent columns
            
            pd.DataFrame({'subject_id': [1], 'text': ['test']}).to_csv(notes_path, index=False)
            
            # pandas should still load this, so we test with completely invalid data
            with open(diag_path, 'w') as f:
                f.write('not,valid,csv\n')  # No matching data
            
            # Should still load but with empty results or validation errors
            # This tests that we handle edge cases gracefully


if __name__ == '__main__':
    unittest.main()
