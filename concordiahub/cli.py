"""
Command line interface for ConcordiaHub, launches the dashboard in a browser
"""


import os
import subprocess
import argparse

from concordiahub import config

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
    parser.add_argument(
        "--data-dir",
        help=f"Directory holding the model data (default: ${config.DATA_DIR_ENV_VAR} "
             f"if set, otherwise ./{config.DEFAULT_DATA_DIR_NAME} in the current directory)"
    )
    args = parser.parse_args()

    # The streamlit app runs in a subprocess, so pass the location by environment
    env = os.environ.copy()
    if args.data_dir:
        env[config.DATA_DIR_ENV_VAR] = os.path.abspath(os.path.expanduser(args.data_dir))
    data_dir = env.get(config.DATA_DIR_ENV_VAR) or os.path.join(os.getcwd(), config.DEFAULT_DATA_DIR_NAME)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Using data directory: {data_dir}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "dashboard", "app.py")
    cmd = ["streamlit", "run", app_path]
    if args.no_browser:
        cmd += ["--server.headless", "true"]

    subprocess.run(cmd, env=env)

if __name__ == "__main__":  # pragma: no cover
    main()
