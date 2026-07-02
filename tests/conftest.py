import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / ".claude" / "skills" / "linearrag" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
