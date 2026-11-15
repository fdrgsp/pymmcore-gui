"""Dispatch the headless worker before importing the GUI command line."""

import sys

STITCH_WORKER_ARG = "_stitch-worker"


def main() -> None:
    """Run the stitching worker if asked to, otherwise the GUI's command line."""
    if len(sys.argv) > 1 and sys.argv[1] == STITCH_WORKER_ARG:
        from pymmcore_gui._stitching._worker import worker_main

        sys.exit(worker_main(sys.argv[2:]))

    from pymmcore_gui._cli import main as cli_main

    cli_main()


__all__ = ["STITCH_WORKER_ARG", "main"]

if __name__ == "__main__":
    main()
