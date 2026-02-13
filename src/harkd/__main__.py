"""Entry point for running harkd as a module: python -m harkd"""

import argparse
import sys

from harkd.server import run_server


def main() -> int:
    """Main entry point for harkd daemon.

    Returns:
        Exit code
    """
    parser = argparse.ArgumentParser(
        prog="harkd",
        description="Voice recording and transcription daemon",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Server host (default: from config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default: from config)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-reload",
    )

    args = parser.parse_args()

    # Determine reload setting
    reload = None
    if args.reload:
        reload = True
    elif args.no_reload:
        reload = False

    try:
        run_server(host=args.host, port=args.port, reload=reload)
        return 0
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
