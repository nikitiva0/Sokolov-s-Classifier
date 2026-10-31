"""Запуск yt-dlp с резервным DNS (net.install_dns_fallback); вызывается только через subprocess."""
import sys
from pathlib import Path

sys.path[0] = str(Path(__file__).resolve().parents[1])  # tools/, а не tools/harvest/
from harvest.net import install_dns_fallback  # noqa: E402

install_dns_fallback()
import yt_dlp  # noqa: E402

sys.exit(yt_dlp.main(sys.argv[1:]))
