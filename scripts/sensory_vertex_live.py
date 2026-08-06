"""Explicit live commands for the sensory-48 Vertex Batch boundary."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.sensory_embedding.vertex_live import (
    LIVE_FLAG,
    VertexLiveError,
    cleanup_run,
    download_outputs,
    refresh_job_status,
    submit_shard_once,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Operate the reviewed sensory Vertex Batch cloud boundary. "
            "Every remote command is blocked unless --execute-live is explicit."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="Upload and create one shard once.")
    create.add_argument(LIVE_FLAG, action="store_true")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--ledger", type=Path, required=True)
    create.add_argument("--shard-index", type=int, required=True)
    create.add_argument("--allow-soft-stop-override", action="store_true")

    status = commands.add_parser("status", help="Read one remote job state once.")
    status.add_argument(LIVE_FLAG, action="store_true")
    status.add_argument("--ledger", type=Path, required=True)
    status.add_argument("--shard-index", type=int, required=True)

    download = commands.add_parser(
        "download",
        help="Download all output objects into create-only verified files.",
    )
    download.add_argument(LIVE_FLAG, action="store_true")
    download.add_argument("--ledger", type=Path, required=True)
    download.add_argument("--shard-index", type=int, required=True)
    download.add_argument("--output-dir", type=Path, required=True)

    cleanup = commands.add_parser(
        "cleanup",
        help="Delete run objects and its dedicated bucket after verified downloads.",
    )
    cleanup.add_argument(LIVE_FLAG, action="store_true")
    cleanup.add_argument("--ledger", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "create":
            result = submit_shard_once(
                execute_live=arguments.execute_live,
                manifest_path=arguments.manifest,
                ledger_path=arguments.ledger,
                shard_index=arguments.shard_index,
                allow_soft_stop_override=arguments.allow_soft_stop_override,
            )
        elif arguments.command == "status":
            result = refresh_job_status(
                execute_live=arguments.execute_live,
                ledger_path=arguments.ledger,
                shard_index=arguments.shard_index,
            )
        elif arguments.command == "download":
            result = download_outputs(
                execute_live=arguments.execute_live,
                ledger_path=arguments.ledger,
                shard_index=arguments.shard_index,
                output_dir=arguments.output_dir,
            )
        else:
            result = cleanup_run(
                execute_live=arguments.execute_live,
                ledger_path=arguments.ledger,
            )
    except VertexLiveError as error:
        print(
            json.dumps(
                {
                    "status": "LIVE_BLOCKED",
                    "phase": error.phase,
                    "message": str(error),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    except Exception:
        print(
            json.dumps(
                {
                    "status": "LIVE_BLOCKED",
                    "phase": "unexpected_failure",
                    "message": "unexpected adapter failure; no details were emitted",
                },
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
