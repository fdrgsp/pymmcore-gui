"""Entry point for `python -m pymmcore_gui`, the `pymm`/`mmgui` scripts and the bundle.

The stitching worker is launched through this same entry point (the frozen app
has no other), so its argument is caught before anything else is imported:
the CLI module pulls in typer and the GUI, and the worker must stay Qt-free.
"""

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
