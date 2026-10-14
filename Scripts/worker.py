"""对话队列 worker 入口。项目根执行：

  python scripts/worker.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.worker import main


if __name__ == "__main__":
    main()
