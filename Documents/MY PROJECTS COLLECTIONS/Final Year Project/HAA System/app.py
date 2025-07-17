"""
Main application file 
This file is used for HuggingFace Spaces deployment
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from gradio_app import create_app

# Create the Gradio interface
demo = create_app()

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=False
    )