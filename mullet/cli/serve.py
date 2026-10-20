"""mullet serve — start the API server."""

import logging


def add_serve_parser(subparsers):
    """Add the serve subcommand to the CLI parser."""
    parser = subparsers.add_parser("serve", help="Start the Mullet API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address")
    parser.add_argument("--port", type=int, default=3000, help="Port number")
    return parser


def run_serve(args):
    """Start the API server."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from mullet.server.app import run_server
    print(f"Starting Mullet server on {args.host}:{args.port}")
    print(f"  REST API: http://{args.host}:{args.port}/v1/chat/completions")
    print(f"  WebSocket: ws://{args.host}:{args.port}/ws")
    print(f"  Health: http://{args.host}:{args.port}/health")
    run_server(host=args.host, port=args.port)
