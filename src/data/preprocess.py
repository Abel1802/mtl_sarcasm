import os
import argparse
import logging
from pathlib import Path

from extract_fea import TextFeatureExtractor, AudioFeatureExtractor, VideoFeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessing.log")
    ]
)
logger = logging.getLogger(__name__)

def validate_paths(paths_dict):
    """Validate that all required paths exist"""
    for name, path in paths_dict.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} path does not exist: {path}")
    logger.info("All paths validated successfully")

def process_text_features(csv_path, output_dir, text_column="transcription", id_column="id"):
    """Process text features using RoBERTa embeddings"""
    logger.info("Starting text feature extraction...")
    
    # Validate paths
    validate_paths({
        "CSV file": csv_path,
        "Output directory": output_dir
    })
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process text features
    extractor = TextFeatureExtractor()
    extractor.process_csv_to_pt(
        csv_path=csv_path,
        text_column=text_column,
        id_column=id_column,
        output_dir=output_dir
    )
    
    logger.info("Text feature extraction completed successfully")

def process_audio_features(csv_path, audio_dir, output_dir):
    """Process audio features from WAV files"""
    logger.info("Starting audio feature extraction...")
    
    # Validate paths
    validate_paths({
        "CSV file": csv_path,
        "Audio directory": audio_dir,
        "Output directory": output_dir
    })
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process audio features
    extractor = AudioFeatureExtractor()
    extractor.process_audio_list_to_pt(
        csv_path=csv_path,
        audio_dir=audio_dir,
        output_dir=output_dir
    )
    
    logger.info("Audio feature extraction completed successfully")

def process_video_features(csv_path, video_dir, output_dir):
    """Process video features from video files"""
    logger.info("Starting video feature extraction...")
    
    # Validate paths
    validate_paths({
        "CSV file": csv_path,
        "Video directory": video_dir,
        "Output directory": output_dir
    })
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process video features
    extractor = VideoFeatureExtractor()
    extractor.process_video_list_to_pt(
        csv_path=csv_path,
        video_dir=video_dir,
        output_dir=output_dir
    )
    
    logger.info("Video feature extraction completed successfully")

def create_parser():
    """Create argument parser for command line interface"""
    parser = argparse.ArgumentParser(
        description="Multimodal Sarcasm Detection - Feature Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
        python preprocess.py --mode text --csv-path data/input.csv --output-dir processed/text
        python preprocess.py --mode audio --csv-path data/input.csv --audio-dir raw/audio --output-dir processed/audio
        python preprocess.py --mode video --csv-path data/input.csv --video-dir raw/video --output-dir processed/video
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["text", "audio", "video", "all"],
        required=True,
        help="Processing mode: text, audio, video, or all"
    )
    
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the filtered CSV file containing metadata"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./processed_data",
        help="Base output directory for processed features (default: ./processed_data)"
    )
    
    parser.add_argument(
        "--text-column",
        type=str,
        default="transcription",
        help="Column name containing text data (default: transcription)"
    )
    
    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Column name containing IDs (default: id)"
    )
    
    parser.add_argument(
        "--audio-dir",
        type=str,
        help="Directory containing audio files (.wav)"
    )
    
    parser.add_argument(
        "--video-dir",
        type=str,
        help="Directory containing video files"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Starting preprocessing in {args.mode} mode")
    logger.info(f"CSV path: {args.csv_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        if args.mode == "text":
            process_text_features(
                csv_path=args.csv_path,
                output_dir=os.path.join(args.output_dir, "text_features"),
                text_column=args.text_column,
                id_column=args.id_column
            )
            
        elif args.mode == "audio":
            if not args.audio_dir:
                raise ValueError("--audio-dir is required for audio processing")
            process_audio_features(
                csv_path=args.csv_path,
                audio_dir=args.audio_dir,
                output_dir=os.path.join(args.output_dir, "audio_features")
            )
            
        elif args.mode == "video":
            if not args.video_dir:
                raise ValueError("--video-dir is required for video processing")
            process_video_features(
                csv_path=args.csv_path,
                video_dir=args.video_dir,
                output_dir=os.path.join(args.output_dir, "video_features")
            )
            
        elif args.mode == "all":
            # Process all modalities
            logger.info("Processing all modalities...")
            
            # Text processing
            process_text_features(
                csv_path=args.csv_path,
                output_dir=os.path.join(args.output_dir, "text_features"),
                text_column=args.text_column,
                id_column=args.id_column
            )
            
            # Audio processing (if audio dir provided)
            if args.audio_dir:
                process_audio_features(
                    csv_path=args.csv_path,
                    audio_dir=args.audio_dir,
                    output_dir=os.path.join(args.output_dir, "audio_features")
                )
            else:
                logger.warning("Skipping audio processing: --audio-dir not provided")
            
            # Video processing (if video dir provided)
            if args.video_dir:
                process_video_features(
                    csv_path=args.csv_path,
                    video_dir=args.video_dir,
                    output_dir=os.path.join(args.output_dir, "video_features")
                )
            else:
                logger.warning("Skipping video processing: --video-dir not provided")
                
        logger.info("All preprocessing tasks completed successfully!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()