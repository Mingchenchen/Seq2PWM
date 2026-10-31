"""Generate the ESM-2 cache used by Seq2PWM."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from seq2pwm.cli.embed import main  # noqa: E402


if __name__ == "__main__":
    main()
