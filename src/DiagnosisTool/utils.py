import json
import os
import datetime
import psutil  # para capturar CPU/memória
from pathlib import Path

class JSONLogger:
    def __init__(self, log_file="logs.jsonl"):
        self.log_file = log_file
        log_dir = os.path.dirname(log_file)
        # Só cria diretório se houver caminho definido
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log(self, service, level, event, details=None, performance=None):
        entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "service": service,
            "level": level,
            "event": event,
            "details": details or {},
            "performance": performance or self._collect_performance()
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def _collect_performance(self):
        return {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage_mb": psutil.virtual_memory().used / (1024 * 1024)
        }
