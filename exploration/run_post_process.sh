ROOT=$(python -c "from dynaconf import Dynaconf; c=Dynaconf(settings_files=['$CONFIG'], merge_enabled=True).default; print(c.logs.root)")
mkdir -p "$ROOT"
python -u post_process.py --config "$CONFIG" 2>&1 | tee "$ROOT/post_process.log"