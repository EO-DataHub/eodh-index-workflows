"""
A static file server with Range support for the fetch-mode tests. It runs in its own process,
because GDAL holds the GIL while it opens a remote file, so a server thread in the test process
could not answer.

    python fixture_server.py <root> <control.json> <requests.log>

It prints its port on the first line of stdout. On every request it re-reads `control.json`:

    {"pages": {path: body}, "failures": {path: [status, ...]}, "hang": [path, ...]}

`pages` replaces a file's body, `failures` lists the statuses to answer to the first requests for a
path, and paths in `hang` never get an answer. Each request's path is appended to `requests.log`.
"""

import json
import re
import sys
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def serve(root: Path, control: Path, log: Path) -> None:
    root = root.resolve()
    seen = Counter()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self._serve(body=False)

        def do_GET(self):
            self._serve(body=True)

        def _empty(self, status):
            self.send_response(status)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _serve(self, body):
            path = self.path.split("?", 1)[0]
            settings = json.loads(control.read_text()) if control.exists() else {}
            with lock:
                count = seen[path]
                seen[path] += 1
                with log.open("a") as out:
                    out.write(path + "\n")
            if path in settings.get("hang", []):
                time.sleep(60)
                return
            failures = settings.get("failures", {}).get(path, [])
            if count < len(failures):
                return self._empty(failures[count])
            if path in settings.get("pages", {}):
                data = settings["pages"][path].encode()
            else:
                file = (root / path.lstrip("/")).resolve()
                if not file.is_file() or root not in file.parents:
                    return self._empty(404)
                data = file.read_bytes()

            start, end, status = 0, len(data) - 1, 200
            match = re.match(r"bytes=(\d+)-(\d*)", self.headers.get("Range", ""))
            if match:
                start = int(match.group(1))
                end = min(int(match.group(2)) if match.group(2) else end, len(data) - 1)
                status = 206
            chunk = data[start : end + 1]
            self.send_response(status)
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.end_headers()
            if body:
                self.wfile.write(chunk)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    print(httpd.server_address[1], flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
