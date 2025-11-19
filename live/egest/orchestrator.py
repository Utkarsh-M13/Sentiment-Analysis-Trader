# live/egest/orchestrator.py
from common.logger import get_logger
from dotenv import load_dotenv
from live.egest.accumulate_and_trade import accumulate_and_trade
from datetime import datetime, time as dt_time
import pytz

log = get_logger("orchestrator_egest")
US_EASTERN = pytz.timezone("US/Eastern")

def is_trading_session_now() -> bool:
    """
    - weekday (Mon–Fri)
    - between 9:30 and 16:00 ET
    """
    now = datetime.now(US_EASTERN)
    if now.weekday() >= 5:  # 5=Sat, 6=Sun
        return False

    start = dt_time(9, 30)
    end = dt_time(16, 0)
    t = now.time()
    return start <= t <= end

def handler(event, context):
    """
    AWS Lambda entrypoint for ingest.
    """
    # If you ever want to override date via the event, you can read it here.
    run_egest()
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

def run_egest():
  if not is_trading_session_now():
        log.info("Not in trading session - skipping accumulate_and_trade().")
        return
  
  load_dotenv()
  log.info("Starting EGEST orchestrator.")
  accumulate_and_trade()
  log.info("Finished EGEST orchestrator.")


if __name__ == "__main__":
    main()
