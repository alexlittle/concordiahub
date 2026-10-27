"""
Command line interface for ConcordiaHub, launches the dashboard in a browser
"""


import os
import subprocess
import argparse
from pathlib import Path

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


# System locations that should never be used to hold model data
PROTECTED_DIRS = ("/bin", "/boot", "/dev", "/etc", "/lib", "/lib64", "/proc", "/sbin", "/sys", "/usr")


def validate_data_dir(path: str) -> str:
    """
    Resolves the data directory to an absolute path and checks it is usable

    Args:
        path: the directory as given by the user (may be relative or start with ~)

    Returns:
        str: the resolved absolute path

    Raises:
        ValueError: if the path is not a directory, or is the filesystem root or a system location
    """
    resolved = Path(path).expanduser().resolve()
    if resolved == Path(resolved.anchor):
        raise ValueError(f"Data directory cannot be the filesystem root: {resolved}")
    for protected in PROTECTED_DIRS:
        if resolved.is_relative_to(protected):
            raise ValueError(f"Data directory cannot be inside a system location ({protected}): {resolved}")
    if resolved.exists() and not resolved.is_dir():
        raise ValueError(f"Data directory is not a directory: {resolved}")
    return str(resolved)


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
    configured = args.data_dir or env.get(config.DATA_DIR_ENV_VAR) or config.DEFAULT_DATA_DIR_NAME
    try:
        data_dir = validate_data_dir(configured)
    except ValueError as err:
        parser.error(str(err))
    env[config.DATA_DIR_ENV_VAR] = data_dir
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
