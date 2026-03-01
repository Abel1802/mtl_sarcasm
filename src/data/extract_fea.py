import os
import numpy as np
import pandas as pd
import torch
import torchaudio
import h5py
from transformers import BertTokenizer, BertModel
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from transformers import VideoMAEImageProcessor, VideoMAEModel
import decord
import logging
from tqdm import tqdm

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
                cls_embedding = torch.from_numpy(cls_embedding)
                # print(f"cls_embedding: {cls_embedding.shape}") (768)
                features.append(cls_embedding)
                
                if (i + 1) % 100 == 0:
                    logging.info(f"Processed {i + 1}/{len(texts)} samples")
        
        return features

    def process_csv_to_pt(self, csv_path, text_column, id_column, output_dir):
        """
        CSV -> Features -> save_dir/*.pt
        """

        os.makedirs(output_dir, exist_ok=True)

        df = self.load_csv_data(csv_path)
        texts = df[text_column].tolist()
        ids = df[id_column].tolist()

        features = self.extract_features(texts)

        for uid, feature in zip(ids, features):
            save_path = os.path.join(output_dir, f"{uid}.pt")
            torch.save(feature, save_path)

        logging.info(f"Completed processing: {csv_path} -> {output_dir}")


class AudioFeatureExtractor():
    '''
    Using Chinese Wav2Vec2 to extract utterance-level audio feature (768-d)

    Input: wav file
    Output: 768-d embedding
    '''

    def __init__(
        self,
        model_name="TencentGameMate/chinese-wav2vec2-base",
        target_sr=16000
    ):
        self.model_name = model_name
        self.target_sr = target_sr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        logging.info(f"Initialized AudioFeatureExtractor with {model_name} on {self.device}")

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

    def load_audio(self, audio_path):
        waveform, sr = torchaudio.load(audio_path)

        # convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)

        return waveform.squeeze(0)  # (T,)

    def extract_feature(self, audio_path):
        waveform = self.load_audio(audio_path)

        inputs = self.processor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values)

        # (1, seq_len, 768)
        hidden_states = outputs.last_hidden_state

        # Mean pooling -> utterance level embedding
        embedding = hidden_states.mean(dim=1).squeeze(0)  # (768,)
        # print(f"audio embedding: {embedding.shape}")

        return embedding.cpu()
    
    def _get_audio_paths(self, audio_dir, csv_path):
        df = self.load_csv_data(csv_path)
        ids = df["id"].tolist()
        audio_paths = [os.path.join(audio_dir, f"{uid}.wav") for uid in ids]
        return ids, audio_paths
    
    def process_audio_list_to_pt(self, csv_path, audio_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        # get ids and audio paths
        ids, audio_paths = self._get_audio_paths(audio_dir, csv_path)

        for i, (audio_path, uid) in enumerate(zip(audio_paths, ids)):
            feature = self.extract_feature(audio_path)
            save_path = os.path.join(output_dir, f"{uid}.pt")
            torch.save(feature, save_path)

            if (i + 1) % 100 == 0:
                logging.info(f"Processed {i + 1}/{len(audio_paths)} samples")

        logging.info(f"Completed audio feature extraction -> {output_dir}")


decord.bridge.set_bridge('native')

class VideoFeatureExtractor:
    """
    High-Performance Standalone VideoMAE Feature Extractor
    Uses `decord` for fast frame fetching without DataLoader overhead.
    """
    def __init__(
        self,
        model_name="MCG-NJU/videomae-base",
        num_frames=16,
        batch_size=16,  # 提速后建议保持 16 或 32
        use_fp16=True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and torch.cuda.is_available()

        # 加载模型
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        logging.info(f"Loaded {model_name} on {self.device}")

    def load_video_decord(self, video_path):
        """
        核心提速点：使用 decord 进行 O(1) 复杂度的按帧提取
        """
        try:
            # 仅读取视频元数据，不解码
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            
            if total_frames == 0:
                raise ValueError("Decoded 0 frames")

            # 生成均匀采样的索引
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            
            # 精准抓取这 16 帧，转换为 list of numpy arrays (H, W, C)
            frames = vr.get_batch(indices).asnumpy()
            return list(frames)
            
        except Exception as e:
            raise RuntimeError(f"Cannot process video: {e}")

    def extract_batch(self, batch_frames):
        """
        批量前向传播
        """
        inputs = self.processor(
            batch_frames,
            return_tensors="pt"
        )

        pixel_values = inputs["pixel_values"].to(self.device)

        with torch.no_grad():
            if self.use_fp16 and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = self.model(pixel_values)
            else:
                outputs = self.model(pixel_values)

        hidden_states = outputs.last_hidden_state  # (B, tokens, 768)

        # 聚合全局特征 (Mean Pooling)
        embedding = hidden_states.mean(dim=1)      # (B, 768)
        return embedding.cpu()

    def process_video_list_to_pt(self, csv_path, video_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        df = pd.read_csv(csv_path)
        
        # 加入断点续传逻辑，如果中途断开，下次运行会自动跳过已生成的 .pt 文件
        ids = [str(uid) for uid in df["id"].tolist() if not os.path.exists(os.path.join(output_dir, f"{uid}.pt"))]

        if not ids:
            logging.info(f"All features already extracted in {output_dir}!")
            return

        batch_frames = []
        batch_ids = []

        progress_bar = tqdm(total=len(ids), desc="Extracting Video Features")

        for uid in ids:
            video_path = os.path.join(video_dir, f"{uid}.mp4")

            if not os.path.exists(video_path):
                logging.warning(f"File not found: {uid}")
                progress_bar.update(1)
                continue

            try:
                # 1. 高效加载视频
                frames = self.load_video_decord(video_path)
                batch_frames.append(frames)
                batch_ids.append(uid)

                # 2. 满 Batch 进行推理
                if len(batch_frames) == self.batch_size:
                    embeddings = self.extract_batch(batch_frames)

                    for emb, vid in zip(embeddings, batch_ids):
                        save_path = os.path.join(output_dir, f"{vid}.pt")
                        torch.save(emb, save_path)
                        progress_bar.update(1)

                    # 3. 清理内存，防止 OOM
                    batch_frames.clear()
                    batch_ids.clear()
                    torch.cuda.empty_cache()

            except Exception as e:
                logging.warning(f"Skip {uid}: {e}")
                progress_bar.update(1)

        # 4. 处理剩余的不满一个 Batch 的数据
        if len(batch_frames) > 0:
            embeddings = self.extract_batch(batch_frames)
            for emb, vid in zip(embeddings, batch_ids):
                save_path = os.path.join(output_dir, f"{vid}.pt")
                torch.save(emb, save_path)
                progress_bar.update(1)
            
            batch_frames.clear()
            batch_ids.clear()
            torch.cuda.empty_cache()

        progress_bar.close()
        logging.info(f"Completed extraction → {output_dir}")
