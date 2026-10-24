"""Keep a loopback SSH review tunnel alive, reconnecting after disconnection."""
import argparse
import socket
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", help="SSH user@host")
    parser.add_argument("--ssh-port", type=int, required=True)
    parser.add_argument("--local-port", type=int, default=18774)
    parser.add_argument("--remote-port", type=int, default=17874)
    args = parser.parse_args()
    # Refuse an occupied port instead of interfering with another process.
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.local_port))
    command = [
        "ssh", "-C", "-NT", "-p", str(args.ssh_port),
        "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
        "-o", "ConnectTimeout=20", "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
        "-L", f"127.0.0.1:{args.local_port}:127.0.0.1:{args.remote_port}",
        args.target,
    ]
    child = None
    try:
        while True:
            print(f"Connecting review tunnel on 127.0.0.1:{args.local_port}", flush=True)
            child = subprocess.Popen(command)
            code = child.wait()
            print(f"Tunnel disconnected ({code}); reconnecting in 3 seconds", flush=True)
            time.sleep(3)
    except KeyboardInterrupt:
        if child is not None and child.poll() is None:
            child.terminate()
            child.wait()


if __name__ == "__main__":
    main()
