"""Serve cached official frontend assets locally; relay API/WS to an SSH tunnel."""
import argparse
import http.client
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import select
import socket
from urllib.parse import urlsplit


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        if self.headers.get("Upgrade", "").lower() == "websocket":
            self.close_connection = True
            with socket.create_connection(("127.0.0.1", BACKEND), timeout=30) as remote:
                request = (f"GET {self.path} HTTP/1.1\r\n" +
                           "".join(f"{k}: {v}\r\n" for k, v in self.headers.items()) + "\r\n").encode()
                remote.sendall(request)
                while True:
                    ready, _, _ = select.select([self.connection, remote], [], [], 30)
                    for source in ready:
                        data = source.recv(65536)
                        if not data:
                            return
                        (remote if source is self.connection else self.connection).sendall(data)
            return
        if urlsplit(self.path).path in {"/object_info", "/api/object_info"} and (ROOT / "object_info.json").is_file():
            self.path = "/object_info.json"
            return super().do_GET()
        if Path(self.translate_path(self.path)).is_file() or self.path == "/":
            return super().do_GET()
        self.proxy()

    def do_HEAD(self):
        if Path(self.translate_path(self.path)).is_file() or self.path == "/":
            return super().do_HEAD()
        self.proxy()

    def do_POST(self):
        self.proxy()

    def do_PUT(self):
        self.proxy()

    def do_DELETE(self):
        self.proxy()

    def proxy(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        headers = {k: v for k, v in self.headers.items() if k.lower() != "connection"}
        connection = http.client.HTTPConnection("127.0.0.1", BACKEND, timeout=90)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
            self.send_response(response.status)
            for key, value in response.getheaders():
                if key.lower() not in {"content-length", "transfer-encoding", "connection"}:
                    self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)) if self.command != "HEAD" else
                             response.getheader("Content-Length", "0"))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(data)
        finally:
            connection.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18778)
    parser.add_argument("--backend-port", type=int, default=18777)
    options = parser.parse_args()
    ROOT, BACKEND = options.static.resolve(), options.backend_port
    if not (ROOT / "index.html").is_file():
        parser.error("The official frontend static directory must contain index.html")
    ThreadingHTTPServer(("127.0.0.1", options.port), Handler).serve_forever()
