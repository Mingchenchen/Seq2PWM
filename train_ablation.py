"""Run the Seq2PWM ablation training workflow."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from seq2pwm.cli.train_ablation import main


if __name__ == "__main__":
    main()
