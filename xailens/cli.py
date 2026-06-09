import os
import subprocess

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(base_dir, "dashboard", "app.py")
    subprocess.run(["streamlit", "run", app_path])

if __name__ == "__main__":
    main()
