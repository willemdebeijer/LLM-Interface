# src/llm_interface/cli.py
import json
import math
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

    def log_message(self, format, *args):
        # Suppress the default server log messages
        pass

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

        if path == "/api/chats":
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
            page = self.path.split("=")[1] if "=" in self.path else "1"
            per_page = 250

            # Get all chat files and sort by timestamp (newest first)
            chat_files = list(Path(self.debug_dir).glob("*.json"))
            chat_files.sort(reverse=True)

            # Calculate pagination
            total_chats = len(chat_files)
            total_pages = math.ceil(total_chats / per_page)
            start_idx = (int(page) - 1) * per_page
            end_idx = start_idx + per_page

            # Get paginated subset of files
            paginated_files = chat_files[start_idx:end_idx]

            chats = []
            for file_path in paginated_files:
                try:
                    with open(file_path, "r") as f:
                        chat = json.load(f)
                        chats.append(chat)
                except Exception as e:
                    print(f"Error loading chat file {file_path}: {e}")

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "chats": chats,
                        "pagination": {
                            "total": total_chats,
                            "per_page": per_page,
                            "current_page": int(page),
                            "total_pages": total_pages,
                        },
                    }
                ).encode()
            )
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

    # Try to use the default port first
    default_port = 7464
    handler = create_handler(debug_dir)

    try:
        httpd = HTTPServer(("localhost", default_port), handler)
        actual_port = default_port
        print(f"Starting server on default port {default_port}")
    except OSError:
        # If default port is not available, let OS choose one
        httpd = HTTPServer(("localhost", 0), handler)
        actual_port = httpd.server_port
        print(f"Default port {default_port} was not available")
        print(f"Using alternative port {actual_port}")

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
