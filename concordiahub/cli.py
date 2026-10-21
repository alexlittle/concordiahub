"""
Command line interface for ConcordiaHub, launches the dashboard in a browser
"""


import os
import subprocess
import argparse

from concordiahub import config

# Streamlit settings the dashboard needs. Streamlit only reads .streamlit/config.toml
# from the current directory, so these are passed as environment variables instead.
# Anything already set in the user's environment takes precedence.
STREAMLIT_DEFAULTS = {
    "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "poll",
    "STREAMLIT_RUNNER_FAST_RERUNS": "true",
    "STREAMLIT_CLIENT_SHOW_SIDEBAR_NAVIGATION": "false",
    "STREAMLIT_CLIENT_TOOLBAR_MODE": "minimal",
    "STREAMLIT_THEME_PRIMARY_COLOR": "#1A6B6B",
}


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
    for key, value in STREAMLIT_DEFAULTS.items():
        env.setdefault(key, value)
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
