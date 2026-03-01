# Instructions

## 🚀 Usage

### Process text features only
python preprocess.py --mode text --csv-path data/labels_filtered.csv --output-dir ./features

### Process audio features only
python preprocess.py --mode audio --csv-path data/labels_filtered.csv --audio-dir your_audio_dir --output-dir ./features

### Process video features only
python src/data/preprocess.py --mode video --csv-path /projects/0/prjs0864/phd_projects/raw_data/sarcasm_zh26/labels_transcriptions_Sheet1_filtered.csv  --video-dir /projects/0/prjs0864/phd_projects/raw_data/sarcasm_zh26/videos_final --output-dir ./processed_data