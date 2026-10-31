"""Run direct Seq2PWM prediction from a repository checkout."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from seq2pwm.cli.predict import main  # noqa: E402


if __name__ == "__main__":
    main()
