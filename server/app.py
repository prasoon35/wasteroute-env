import sys
import os

# Add parent directory to path so 'models' is always findable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, '/app/env')

from openenv.core.env_server.http_server import create_app
from models import WasteAction, WasteObservation
from server.wasteroute_env_environment import WasteRouteEnvironment

app = create_app(
    WasteRouteEnvironment,
    WasteAction,
    WasteObservation,
    env_name="wasteroute_env",
    max_concurrent_envs=10,
)

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "healthy"}


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)
