# src/llm_interface/cli.py
import json
import os
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import pkg_resources


class LLMViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, debug_dir: str, **kwargs):
        self.debug_dir = debug_dir
        super().__init__(*args, **kwargs)

    def do_GET(self):
        # Parse the URL path
        parsed_path = urlparse(self.path)
        path = parsed_path.path

        if path == "/":
            # Redirect root to viewer.html
            self.send_response(302)
            self.send_header("Location", "/static/viewer.html")
            self.end_headers()
            return

        if "static" in path:
            # Get the filename from the path
            filename = path.split("/")[-1]
            # Get the file extension
            ext = os.path.splitext(filename)[1]
            # Map extensions to content types
            content_types = {
                ".html": "text/html",
                ".css": "text/css",
                ".js": "application/javascript",
            }
            content_type = content_types.get(ext, "text/plain")

            # Get the file path from the package's static directory
            file_path = pkg_resources.resource_filename(
                "llm_interface", path.lstrip("/")
            )
            self.serve_file(file_path, content_type=content_type)
            return

        if path == "/api/calls":
            # Serve the JSON data from the debug directory
            self.serve_calls()
            return

        # Default: return 404
        self.send_error(404)

    def serve_file(self, filepath, content_type):
        try:
            with open(filepath, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            print(f"Error serving file: {e}")
            self.send_error(500)

    def serve_calls(self):
        try:
            calls = []
            for file in Path(self.debug_dir).glob("*.json"):
                with open(file) as f:
                    calls.append(json.load(f))

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(calls).encode())
        except Exception as e:
            print(f"Error serving calls: {e}")
            self.send_error(500)


def create_handler(debug_dir):
    return lambda *args, **kwargs: LLMViewerHandler(
        *args, debug_dir=debug_dir, **kwargs
    )


def main():
    # Get the current working directory
    cwd = os.getcwd()
    debug_dir: str = os.path.join(cwd, ".llm_recorder")

    if not os.path.exists(debug_dir):
        print(f"No debug data found in {debug_dir}")
        print("Start using LLMInterface with debug_mode=True first.")
        return

    # Start server on a random available port
    port = 0  # This tells the OS to pick an available port
    handler = create_handler(debug_dir)
    httpd = HTTPServer(("localhost", port), handler)
    actual_port = httpd.server_port

    # Start the server in a separate thread
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()

    url = f"http://localhost:{actual_port}/static/viewer.html"
    print(f"Opening LLM Interface at {url}")
    print("Press Ctrl+C to stop the server")
    webbrowser.open(url)

    try:
        # Keep the main thread alive
        thread.join()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
