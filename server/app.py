# import sys
# import os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.insert(0, '/app/env')

# from openenv.core.env_server.http_server import create_app
# from models import WasteAction, WasteObservation
# from server.wasteroute_env_environment import WasteRouteEnvironment

# app = create_app(
#     WasteRouteEnvironment,
#     WasteAction,
#     WasteObservation,
#     env_name="wasteroute_env",
#     max_concurrent_envs=10,
# )

# # HF Spaces health check
# from fastapi import FastAPI
# @app.get("/")
# def root():
#     return {"status": "healthy"}

# @app.get("/health")
# def health():
#     return {"status": "healthy"}

# def main(host: str = "0.0.0.0", port: int = 7860):
#     import uvicorn
#     uvicorn.run(app, host=host, port=port)

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--port", type=int, default=7860)
#     args = parser.parse_args()
#     main(port=args.port)




import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/app/env')

from openenv.core.env_server.http_server import create_app

try:
    from ..models import WasteAction, WasteObservation
except ImportError:
    from models import WasteAction, WasteObservation

try:
    from .wasteroute_env_environment import WasteRouteEnvironment
except ImportError:
    from server.wasteroute_env_environment import WasteRouteEnvironment

app = create_app(
    WasteRouteEnvironment,
    WasteAction,
    WasteObservation,
    env_name="wasteroute_env",
    max_concurrent_envs=10,
)

def main() -> None:
    """Entry point for OpenEnv validator (no-arg main)."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == '__main__':
    main()