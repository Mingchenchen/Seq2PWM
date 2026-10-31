"""Run the standard Seq2PWM training workflow."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from seq2pwm.cli.train import main


if __name__ == "__main__":
    main()
