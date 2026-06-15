import os
import sys

# Add the parent directory to sys.path to enable loading src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import app
