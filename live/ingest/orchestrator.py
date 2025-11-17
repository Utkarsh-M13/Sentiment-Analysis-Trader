# live/ingest/orchestrator.py
from common.logger import get_logger
from live.ingest.pull_and_process import pull_and_process_data
from dotenv import load_dotenv

log = get_logger("orchestrator_ingest")


def main():
  load_dotenv()
  log.info("Starting INGEST orchestrator.")
  pull_and_process_data()
  log.info("Finished INGEST orchestrator.")


if __name__ == "__main__":
    main()
