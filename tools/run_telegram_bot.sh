#!/bin/sh
# Telegram-бот в фоне: tools/run_telegram_bot.sh start|stop|status|restart [аргументы telegram_bot.py]
# Журнал: work/telegram_bot.log, PID: work/telegram_bot.pid. Токен: TELEGRAM_BOT_TOKEN или work/telegram_token.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
PID_FILE="$ROOT/work/telegram_bot.pid"
LOG_FILE="$ROOT/work/telegram_bot.log"
RUN_DIR=${SOKOLOV_RUN_DIR:-outputs/runs/now-v3-eq}
DEVICE=${SOKOLOV_DEVICE:-mps}

find_python() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  for candidate in "$ROOT/.venv/bin/python" "$ROOT/../../.venv/bin/python"; do
    if [ -x "$candidate" ]; then echo "$candidate"; return; fi
  done
  echo python3
}

running() {
  [ -f "$PID_FILE" ] || return 1
  pid=$(cat "$PID_FILE")
  ps -p "$pid" -o command= 2>/dev/null | grep -q 'telegram_bot.py'
}

case "${1:-}" in
  start)
    shift
    if running; then echo "Бот уже запущен (PID $(cat "$PID_FILE"))."; exit 0; fi
    mkdir -p "$ROOT/work"
    umask 077
    PY=$(find_python)
    cd "$ROOT"
    PYTHONPATH=src nohup "$PY" tools/telegram_bot.py --run-dir "$RUN_DIR" --device "$DEVICE" "$@" \
      >>"$LOG_FILE" 2>&1 </dev/null &
    echo $! >"$PID_FILE"
    sleep 3
    if running; then
      echo "Бот запущен: PID $(cat "$PID_FILE"), журнал $LOG_FILE"
    else
      rm -f "$PID_FILE"
      echo "Бот завершился сразу. Последние строки журнала:"; tail -n 5 "$LOG_FILE"; exit 1
    fi
    ;;
  stop)
    if ! running; then rm -f "$PID_FILE"; echo "Бот не запущен."; exit 0; fi
    pid=$(cat "$PID_FILE")
    kill "$pid"
    i=0
    while kill -0 "$pid" 2>/dev/null && [ "$i" -lt 20 ]; do sleep 0.5; i=$((i + 1)); done
    if kill -0 "$pid" 2>/dev/null; then kill -9 "$pid"; fi
    rm -f "$PID_FILE"
    echo "Бот остановлен (PID $pid)."
    ;;
  status)
    if running; then echo "Бот работает (PID $(cat "$PID_FILE"))."; else echo "Бот не запущен."; exit 1; fi
    ;;
  restart)
    shift
    "$0" stop
    "$0" start "$@"
    ;;
  *)
    echo "Использование: $0 start|stop|status|restart [аргументы telegram_bot.py]" >&2
    exit 2
    ;;
esac
