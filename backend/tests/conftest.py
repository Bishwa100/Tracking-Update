"""Put the backend root on sys.path so `import app...` works under pytest."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
