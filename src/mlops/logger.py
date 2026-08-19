"""Project-wide logger. File logging is best-effort — see the comment below."""

import logging
import os
import sys
from pathlib import Path

LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s:%(lineno)d - %(message)s"

handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

# A container running as a non-root user may not be able to create this
# directory. A logger that raises on import takes the application down with it,
# so the file handler is optional and stdout is the one that must always work.
try:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(LOG_DIR / "running.log"))
except OSError:
    pass

logging.basicConfig(level=logging.INFO, format=FORMAT, handlers=handlers)

# MLflow logs a great deal at INFO and drowns the pipeline's own output.
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("mlops")
