"""Run the full Seq2PWM evaluation and plotting workflow."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from seq2pwm.cli.infer import main  # noqa: E402


if __name__ == "__main__":
    main()
