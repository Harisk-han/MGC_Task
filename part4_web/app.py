import sys
from pathlib import Path

# Forward to the master multi-navigation application at workspace root
master_app = Path(__file__).parent.parent / "app.py"
sys.path.insert(0, str(master_app.parent))

with open(master_app, encoding="utf-8") as f:
    code = compile(f.read(), str(master_app), "exec")
    exec(code, {"__file__": str(master_app), "__name__": "__main__"})
