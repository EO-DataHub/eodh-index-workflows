"""
A static file server with Range support for the fetch-mode tests. It runs in its own process,
because GDAL holds the GIL while it opens a remote file, so a server thread in the test process
could not answer.

    python fixture_server.py <root> <control.json> <requests.log>

It prints its port on the first line of stdout. On every request it re-reads `control.json`:

    {"pages": {path: body}, "failures": {path: [status, ...]}, "retry_after": {path: seconds},
     "hang": [path, ...], "hang_first": {path: n}, "drop": [path, ...]}

`pages` replaces a file's body. A `pages` key with a query string (`/search?page=2`) answers only
that query, before a key with the bare path. `failures` lists the statuses to answer to the first requests for a
path, and `retry_after` adds a Retry-After header to those failures. Paths in `hang` never get an
answer, and the first `n` requests for a path in `hang_first` get none. For paths in `drop` the
server sends the headers and part of the body, then closes the connection. Each request's path is
appended to `requests.log`, and its path with the query string to `requests.log.full`.
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

        def _empty(self, status, headers=()):
            self.send_response(status)
            for name, value in headers:
                self.send_header(name, value)
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
                with log.with_name(log.name + ".full").open("a") as out:
                    out.write(self.path + "\n")
            if path in settings.get("hang", []) or count < settings.get("hang_first", {}).get(path, 0):
                time.sleep(60)
                return
            failures = settings.get("failures", {}).get(path, [])
            if count < len(failures):
                retry_after = settings.get("retry_after", {}).get(path)
                headers = [("Retry-After", str(retry_after))] if retry_after is not None else []
                return self._empty(failures[count], headers)
            pages = settings.get("pages", {})
            if self.path in pages:
                data = pages[self.path].encode()
            elif path in pages:
                data = pages[path].encode()
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
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            if body and path in settings.get("drop", []):
                self.send_header("Content-Length", str(len(chunk) + 10000))
                self.end_headers()
                self.wfile.write(chunk[: len(chunk) // 2])
                self.wfile.flush()
                self.close_connection = True
                return
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            if body:
                self.wfile.write(chunk)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    print(httpd.server_address[1], flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    serve(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
