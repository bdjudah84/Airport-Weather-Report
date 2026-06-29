"""Command-line entry point.

    python -m weather_accuracy daily         # the once-a-day run (use this in cron)
    python -m weather_accuracy pull          # capture today's forecasts only
    python -m weather_accuracy verify        # score yesterday only
    python -m weather_accuracy verify 2026-06-28
    python -m weather_accuracy report        # rebuild the HTML report only
    python -m weather_accuracy scoreboard    # print the ranking to the terminal

`daily` scores yesterday, captures today, and rewrites the HTML report -- the
whole cycle in one command. Point your scheduler at it once each morning.
"""
from __future__ import annotations

import sys
from datetime import date

import pandas as pd

from . import build_scoreboard, run_forecast_pull, run_verification
from .report import write_report


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    cmd = argv[0]

    if cmd == "daily":
        run_verification()        # score yesterday (now that the day is complete)
        run_forecast_pull()       # capture today's forecasts
        write_report()            # refresh the report page
    elif cmd == "pull":
        run_forecast_pull()
    elif cmd == "verify":
        target = date.fromisoformat(argv[1]) if len(argv) > 1 else None
        run_verification(target)
    elif cmd == "report":
        write_report()
    elif cmd == "scoreboard":
        board = build_scoreboard()
        if board.empty:
            print("No verified forecasts yet -- run 'pull' then 'verify' first.")
        else:
            pd.set_option("display.width", 160, "display.max_columns", 20)
            print(board.round(2).to_string(index=False))
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
