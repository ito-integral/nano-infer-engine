from .app import create_app
from .runtime import build_pd_runtime


app = create_app(build_pd_runtime)
