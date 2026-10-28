"""Serve the power quality analyzer front end with Flask."""

from pathlib import Path

from flask import Flask


frontend_folder = Path(__file__).resolve().parent.parent / "FrontEnd"

app = Flask(
    __name__,
    static_folder=str(frontend_folder),
    static_url_path="",
)


@app.get("/")
def home():
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
