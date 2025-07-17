"""
Main application file 
This file is used for HuggingFace Spaces deployment
"""
"""
Main application file 
This file is used for HuggingFace Spaces deployment
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))  # Ensure current folder is importable

from gradio_app import create_app  # gradio_app.py must be in the same directory

# Create the Gradio interface
app = create_app()

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    )
