"""
Command line interface for xailens, launches the dashboard in a browser
"""


import os
import subprocess
import argparse

def main()-> None:
    """
    Run the streamlit app
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Run Streamlit without opening a browser"
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "dashboard", "app.py")
    cmd = ["streamlit", "run", app_path]
    if args.no_browser:
        cmd += ["--server.headless", "true"]

    subprocess.run(cmd)

if __name__ == "__main__":
    main()
