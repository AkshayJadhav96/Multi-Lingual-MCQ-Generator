import os
import subprocess
import sys
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Define directory paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PREPROCESSING_SCRIPT = os.path.join(SCRIPT_DIR, "data_preprocessing.py")
TRANSFORMER_FINETUNE_SCRIPT = os.path.join(SCRIPT_DIR, "transformer_finetune.py")
GENERATE_MCQ_SCRIPT = os.path.join(SCRIPT_DIR, "generate_mcq.py")

# Function to run a Python script
def run_script(script_path, script_name):
    """Run a Python script and handle errors."""
    logger.info(f"Starting {script_name}...")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"{script_name} completed in {time.time() - start_time:.2f} seconds")
        logger.debug(f"{script_name} output:\n{result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in {script_name}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error in {script_name}: {str(e)}")
        return False

# Main pipeline
def main():
    """Run the full pipeline."""
    logger.info("Starting pipeline execution")
    
    # Step 1: Data Preprocessing
    if not run_script(DATA_PREPROCESSING_SCRIPT, "data_preprocessing.py"):
        logger.error("Pipeline aborted due to preprocessing failure")
        sys.exit(1)
    
    # Step 2: Transformer Fine-tuning
    if not run_script(TRANSFORMER_FINETUNE_SCRIPT, "transformer_finetune.py"):
        logger.error("Pipeline aborted due to fine-tuning failure")
        sys.exit(1)
    
    # Step 3: MCQ Generation
    if not run_script(GENERATE_MCQ_SCRIPT, "generate_mcq.py"):
        logger.error("MCQ generation failed, but pipeline will continue")
    
    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()