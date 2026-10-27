"""Minimal HTTP server around the Lambda handler, for running the embedder outside AWS.

    python scripts/serve.py --port 8080
    curl -XPOST localhost:8080/embed -d '{"images": [{"id": "1", "url": "https://picsum.photos/256"}]}'
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.handler import handler  # noqa: E402

MAX_BODY_BYTES = 64 * 1024 * 1024
# ONNX Runtime already uses all cores per request; running many at once only adds memory pressure
_inference_slots = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_REQUESTS", 2)))


class EmbedHandler(BaseHTTPRequestHandler):
    timeout = 30  # seconds; stops clients that announce a body and never send it
    def _send(self, status, payload):
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not_found"})

    def do_POST(self):
        if self.path not in ("/", "/embed"):
            self._send(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0:
            self._send(400, {"error": "bad_input", "details": "Invalid Content-Length"})
            return
        if length > MAX_BODY_BYTES:
            self._send(413, {"error": "too_large"})
            return
        body = self.rfile.read(length).decode("utf-8", errors="replace")
        try:
            with _inference_slots:
                resp = handler({"body": body}, None)
        except Exception:
            self._send(500, {"error": "internal_error"})
            return
        self._send(resp["statusCode"], resp["body"])


def main():
    parser = argparse.ArgumentParser(description="Serve the SigLIP embedder over HTTP")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8080)))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), EmbedHandler)
    print(f"Listening on http://{args.host}:{args.port}  (POST /embed, GET /health)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
