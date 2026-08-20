"""
Entry point for Hugging Face Spaces (Streamlit SDK looks for app.py at the
repo root by default). This just delegates to the real app.
"""
import runpy
import os

runpy.run_path(os.path.join("app", "streamlit_app.py"), run_name="__main__")
