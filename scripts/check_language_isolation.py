"""CPU smoke test for language/mathematics checkpoint separation.

Run from the repository root: .venv/bin/python scripts/check_language_isolation.py
"""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

# Allow direct execution from scripts/ without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nord_brain.runtime import Brain
from nord_brain.linguistic_training import evaluate


def main():
    with TemporaryDirectory(prefix="nord-isolation-") as tmp:
        brain = Brain(Path(tmp), start_autonomy=False)
        try:
            assert brain.core is not brain.language_core
            assert brain.core.path != brain.language_core.path
            assert brain.core.model.config.layers == brain.language_core.model.config.layers
            assert brain.language_core.path.exists(), "Missing language migration checkpoint"
            result = evaluate(brain)
            assert result["language_loss"] is not None
            original = brain.language_core.steps
            brain.core.train("Quanto é 2+2?", "4", epochs=1)
            assert brain.language_core.steps == original, "Math altered language steps"
            print("OK: checkpoints independent; language evaluation available.")
        finally:
            brain.autonomy.stop()


if __name__ == "__main__":
    main()
