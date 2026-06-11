"""Entry point desktop (PyInstaller) untuk Zaneva ImageKit.

Menjalankan server Flask di port lokal yang kosong, lalu otomatis
membuka browser ketika server sudah siap menerima koneksi.
"""
import os
import socket
import threading
import time
import webbrowser


def find_free_port(start: int = 5000, tries: int = 20) -> int:
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Tidak ada port kosong di rentang {start}-{start + tries - 1}")


def open_browser_when_ready(port: int):
    url = f"http://127.0.0.1:{port}"
    # Startup pertama bisa lama (download model ~900MB), jadi tunggu sabar.
    for _ in range(7200):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.5)
    webbrowser.open(url)


def main():
    port = find_free_port(int(os.environ.get("PORT", 5000)))
    threading.Thread(target=open_browser_when_ready, args=(port,), daemon=True).start()

    print("=" * 50)
    print("  Zaneva ImageKit — Desktop")
    print(f"  Browser akan terbuka di http://127.0.0.1:{port}")
    print("  Model BG di-download saat pertama kali dipakai")
    print("  (isnet ~170MB; birefnet ~900MB bila dipilih).")
    print("  JANGAN tutup jendela ini selama app dipakai.")
    print("=" * 50)

    from app import app  # import memicu preload model BG remover

    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()
