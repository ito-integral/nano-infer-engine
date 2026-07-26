from .app import create_app
from .runtime import build_default_runtime


app = create_app(build_default_runtime)
