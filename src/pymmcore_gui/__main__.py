from pymmcore_gui._cli import main

__all__ = ["main"]

if __name__ == "__main__":
    # Must run before anything else in the frozen (PyInstaller) app: a
    # spawned child process (e.g. a Smart Microscopy analysis worker)
    # re-executes this entry point, and freeze_support() hands it over to
    # multiprocessing instead of launching a second GUI. No-op otherwise.
    import multiprocessing

    multiprocessing.freeze_support()
    main()
