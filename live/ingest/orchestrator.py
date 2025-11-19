# live/ingest/orchestrator.py
from common.logger import get_logger
from live.ingest.pull_and_process import pull_and_process_data
from dotenv import load_dotenv
from datetime import datetime

log = get_logger("orchestrator_ingest")


def run_ingest():
  load_dotenv()
  log.info("Starting INGEST orchestrator.")
  pull_and_process_data()
  log.info("Finished INGEST orchestrator.")


def handler(event, context):
    """
    AWS Lambda entrypoint for ingest.
    """
    # If you ever want to override date via the event, you can read it here.
    run_ingest()
    return {"status": "ok", "timestamp": datetime.now().isoformat()}