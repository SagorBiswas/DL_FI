
import logging, sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterable, Sequence


project_home = Path(__file__).resolve().parent.parent


# ────────────────────────────────────────────────────────────────
def get_configurable_logger(
    name: str,
    *,
    log_dir: str | Path = "logs",
    console_levels: Sequence[int] | Iterable[str] = ("INFO", "WARNING", "ERROR"),
    max_mb: int = 5,
    backups: int = 2,
) -> logging.Logger:
    """
    Build (or fetch) a logger that:
      • writes ALL levels to <log_dir>/<name>.log (rotating)
      • prints only the `console_levels` you supply

    `console_levels` can be e.g. ("INFO", "ERROR") or (logging.INFO, logging.ERROR)
    """

    # ----- sanitise levels --------------------------------------------------
    lvl_map = {lvl for lvl in (
        (l if isinstance(l, int) else logging.getLevelName(l))  # .upper())
        for l in console_levels
    )}
    # convert strings to numeric values
    lvl_nums = {l if isinstance(l, int) else logging.getLevelName(l) for l in lvl_map}
    lvl_nums = {l if isinstance(l, int) else logging._nameToLevel[l] for l in lvl_nums}

    # ----- build / reuse logger --------------------------------------------
    project_home = Path(__file__).resolve().parent.parent
    log_dir = project_home / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:          # already configured
        return logger

    logger.setLevel(logging.DEBUG)         # capture everything

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s: %(message)s"
    dt  = "%Y-%m-%d %H:%M:%S"

    # ---------- file handler ------------------------------------------------
    fh = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backups,
        encoding="utf-8",
    )
    fh.setFormatter(logging.Formatter(fmt, dt))
    fh.setLevel(logging.DEBUG)             # keep all
    logger.addHandler(fh)

    # ---------- console handler with filter --------------------------------
    class LevelFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return record.levelno in lvl_nums

    ch = logging.StreamHandler(sys.stderr)
    ch.setFormatter(logging.Formatter(fmt, dt))
    ch.setLevel(min(lvl_nums))             # need a numeric level, keep filter for exact set
    ch.addFilter(LevelFilter())
    logger.addHandler(ch)

    logger.propagate = False
    logger.info("Logger initialised (console levels: %s)", ", ".join(
        logging.getLevelName(l) for l in sorted(lvl_nums)))
    return logger



"""
############################# Example usage #############################


from logger_utils import get_configurable_logger
import random

logger = get_configurable_logger(
    "training",
    console_levels=("INFO", "WARNING", "ERROR"),   # change as you like
    log_dir="runs",
)

# emit some messages
logger.debug  ("batch %d loss %.3f",  10, random.random())
logger.info   ("epoch %d started",    1)
logger.warning("learning rate plateaued")
logger.error  ("unexpected NaN encountered")


"""