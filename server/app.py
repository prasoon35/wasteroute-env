try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv is required. Install with: uv sync") from e

try:
    from ..models import WasteAction, WasteObservation
    from .wasteroute_env_environment import WasteRouteEnvironment
except ModuleNotFoundError:
    from models import WasteAction, WasteObservation
    from server.wasteroute_env_environment import WasteRouteEnvironment

app = create_app(
    WasteRouteEnvironment,
    WasteAction,
    WasteObservation,
    env_name="wasteroute_env",
    max_concurrent_envs=10,
)

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
