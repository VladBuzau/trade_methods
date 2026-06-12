"""
ai_agent/monitoring/daily_check.py — runner zilnic care orchestreaza:
  1. equity snapshot + kill switch check
  2. predictions log evaluation
  3. trigger auto_retrain daca necesar

Rulat zilnic prin cron sau Task Scheduler:
    python -m ai_agent.monitoring.daily_check
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info(f"DAILY CHECK — {datetime.now(timezone.utc).isoformat()}")
    log.info("=" * 60)

    # 1. Kill switch
    log.info("[1/3] Kill switch check...")
    try:
        from ai_agent.monitoring.kill_switch import check_drawdown
        r = check_drawdown()
        log.info(f"  DD: {r.get('dd_pct', 'N/A')}% — OK={r['ok']}")
    except Exception as e:
        log.error(f"  Kill switch error: {e}")

    # 2. Predictions report
    log.info("[2/3] Predictions accuracy report...")
    try:
        from ai_agent.monitoring.predictions_log import print_report
        print_report(days_back=7)
    except Exception as e:
        log.error(f"  Predictions error: {e}")

    # 3. Auto-retrain (lazy — daca lista de modele e mica)
    log.info("[3/3] Auto-retrain check...")
    try:
        from ai_agent.monitoring.auto_retrain import run_all
        # In productie: ai_agent.config.SYMBOLS; aici sample
        results = run_all(symbols=["EURUSD", "GBPUSD", "XAUUSD"],
                          timeframes=["H1", "H4"],
                          horizons=[24])
        retrained = sum(1 for r in results if r["action"] == "retrained")
        skipped   = sum(1 for r in results if r["action"] == "skipped")
        failed    = sum(1 for r in results if r["action"] in ("failed", "error"))
        log.info(f"  Modele: {retrained} retrained, {skipped} skipped, {failed} failed")
    except Exception as e:
        log.error(f"  Auto-retrain error: {e}")

    log.info("DAILY CHECK COMPLETE")


if __name__ == "__main__":
    main()
