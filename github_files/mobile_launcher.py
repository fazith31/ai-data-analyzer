"""
AI Data Analyzer - Real Mobile App Launcher
Starts Django server on 0.0.0.0:8000 and connects to a secure HTTPS tunnel
so Android & iPhone can install AI Data Analyzer as a REAL NATIVE APP.
"""
import os
import sys
import time
import socket
import subprocess
import shutil
import webbrowser

current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir).lower() == "dataanalyzer":
    DATAANALYZER_DIR = current_dir
    BASE_DIR = os.path.dirname(current_dir)
else:
    BASE_DIR = current_dir
    DATAANALYZER_DIR = os.path.join(BASE_DIR, "dataanalyzer")

MANAGE_PY = os.path.join(DATAANALYZER_DIR, "manage.py")
VENV_PYTHON = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
PYTHON_EXEC = VENV_PYTHON if os.path.exists(VENV_PYTHON) else sys.executable

def is_port_open(port=8000):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0

def start_django_server():
    if is_port_open(8000):
        print("  [OK] Django server is active on port 8000.")
        return None
    
    print("  [*] Starting Django server on 0.0.0.0:8000...")
    proc = subprocess.Popen(
        [PYTHON_EXEC, MANAGE_PY, "runserver", "0.0.0.0:8000"],
        cwd=DATAANALYZER_DIR,
        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
    )
    
    # Wait for server to bind
    for _ in range(15):
        time.sleep(0.5)
        if is_port_open(8000):
            print("  [OK] Django server started successfully!")
            return proc
    print("  [*] Server started in background console.")
    return proc

def run_tunnel():
    ssh_cmd = shutil.which("ssh") or r"C:\Windows\System32\OpenSSH\ssh.exe"
    if not os.path.exists(ssh_cmd) and not shutil.which("ssh"):
        print("  [!] OpenSSH not found on system.")
        return

    print("  [*] Creating secure HTTPS tunnel...")
    print("      (Required by Google Chrome to install as a REAL APP)\n")

    cmd = [
        ssh_cmd,
        "-o", "StrictHostKeyChecking=no",
        "-o", "ServerAliveInterval=30",
        "-R", "80:localhost:8000",
        "nokey@localhost.run"
    ]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        url_found = False
        for line in iter(proc.stdout.readline, ''):
            clean_line = line.strip()
            # Detect public https link from localhost.run
            if ("lhr.life" in clean_line or "localhost.run" in clean_line) and "https://" in clean_line and not url_found:
                for token in clean_line.split():
                    token = token.strip(" ,\"'<>")
                    if token.startswith("https://") and "lhr.life" in token:
                        url_found = True
                        print()
                        print("=" * 70)
                        print("  🎉 YOUR REAL MOBILE APP LINK IS READY!")
                        print("=" * 70)
                        print(f"\n  👉 Open this link on your phone in Chrome / Safari:")
                        print(f"     {token}\n")
                        print("  📲 A QR code is opening on your screen for instant scanning.")
                        print("  📲 When the page opens on your phone, tap '📲 Install App'")
                        print("     to install the real app with home screen icon!")
                        print("=" * 70)
                        print()

                        # Open QR code image in PC browser for easy scanning
                        try:
                            qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=350x350&data={token}"
                            webbrowser.open(qr_url)
                        except Exception:
                            pass
                        break

            print("  " + line, end="")

        proc.wait()
    except KeyboardInterrupt:
        print("\n  [!] Tunnel stopped.")
    except Exception as e:
        print(f"\n  [!] Tunnel error: {e}")

def main():
    os.system("cls" if os.name == "nt" else "clear")
    print("=" * 70)
    print("       📱 AI DATA ANALYZER - REAL MOBILE APP LAUNCHER")
    print("=" * 70)
    print()

    # 1. Start Django
    server_proc = start_django_server()

    print()
    print("=" * 70)
    print("  WHY HTTPS IS REQUIRED FOR A 'REAL APP':")
    print("  On plain HTTP (http://10.61.255.102:8000), Android only allows")
    print("  saving a browser bookmark, NOT a real app.")
    print("  Over HTTPS, Android installs the REAL APP with an app drawer icon!")
    print("=" * 70)
    print()

    # 2. Run the tunnel
    run_tunnel()

    input("\nPress Enter to exit...")

if __name__ == "__main__":
    main()
