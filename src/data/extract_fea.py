import os
import numpy as np
import pandas as pd
import torch
import h5py
from transformers import BertTokenizer, BertModel
import logging

class TextFeatureExtractor():
    ''' Using RoBERTa-wwm-ext extract text feature(text -> 768)
    Input: csv file
    Output: h5 file
    '''
    
    def __init__(self, model_name='hfl/chinese-roberta-wwm-ext', max_length=512):
        """
        Initialize the text feature extractor
        
        Args:
            model_name (str): Pretrained model name
            max_length (int): Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logging.info(f"Initialized TextFeatureExtractor with {model_name} on {self.device}")
    
    def load_csv_data(self, csv_path):
        """
        Load text data from CSV file
        
        Args:
            csv_path (str): Path to CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        try:
            df = pd.read_csv(csv_path)
            logging.info(f"Loaded {len(df)} samples from {csv_path}")
            return df
        except Exception as e:
            logging.error(f"Error loading CSV file: {e}")
            raise
    
    def extract_features(self, texts):
        """
        Extract features from text using RoBERTa
        
        Args:
            texts (list): List of text strings
            
        Returns:
            numpy.ndarray: Extracted features (n_samples, 768)
        """
        features = []
        
        with torch.no_grad():
            for i, text in enumerate(texts):
                # Tokenize text
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Extract features
                outputs = self.model(**inputs)
                # Use CLS token representation (first token)
                cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
                print(f"cls_embedding: {cls_embedding.shape}")
                features.append(cls_embedding)
                
                if (i + 1) % 100 == 0:
                    logging.info(f"Processed {i + 1}/{len(texts)} samples")
        
        return np.array(features)
    
    def save_features(self, features, ids, output_path):
        """
        Save extracted features to HDF5 file
        
        Args:
            features (numpy.ndarray): Extracted features
            ids (list): Sample IDs
            output_path (str): Output HDF5 file path
        """
        try:
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('features', data=features, dtype='float32')
                f.create_dataset('ids', data=np.array(ids, dtype='S'))
            
            logging.info(f"Saved features to {output_path}")
        except Exception as e:
            logging.error(f"Error saving features: {e}")
            raise
    
    def process_csv_to_h5(self, csv_path, text_column, id_column, output_path):
        """
        Complete pipeline: CSV -> Features -> H5
        
        Args:
            csv_path (str): Input CSV file path
            text_column (str): Column name containing text
            id_column (str): Column name containing IDs
            output_path (str): Output HDF5 file path
        """
        # Load data
        df = self.load_csv_data(csv_path)
        
        # Extract texts and IDs
        texts = df[text_column].tolist()
        ids = df[id_column].tolist()
        
        # Extract features
        features = self.extract_features(texts)
        
        # Save to H5
        self.save_features(features, ids, output_path)
        
        logging.info(f"Completed processing: {csv_path} -> {output_path}")

# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize extractor
    extractor = TextFeatureExtractor()
    
    # Process sample data
    extractor.process_csv_to_h5(
        csv_path="data/sample_texts.csv",
        text_column="transcription",
        id_column="id",
        output_path="/projects/0/prjs0864/phd_projects/mlt_sarcasm/processed_data/text_features.h5"
    )



# class AudioFeatureExtractor():
#     '''
#     '''
#     pass


# class VideoFeatureExtractor():
#     '''
#     '''
#     pass