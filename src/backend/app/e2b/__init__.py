"""E2B sandbox and execution utilities."""

from .sandbox import (
    new_persistent_sandbox,
    list_running_or_paused,
    pretty_sbx_info,
    kill_by_id,
    kill_all_running,
    PERSIST_TIMEOUT_SECONDS
)

from .execution import (
    run_in_e2b,
    summarize_execution,
    extract_output_from_execution,
    failed
)

from .artifacts import (
    list_tree,
    print_tree,
    download_all_as_tar,
    download_folder_recursive,
    FileEntry
)

__all__ = [
    # Sandbox management
    "new_persistent_sandbox",
    "list_running_or_paused",
    "pretty_sbx_info",
    "kill_by_id",
    "kill_all_running",
    "PERSIST_TIMEOUT_SECONDS",

    # Code execution
    "run_in_e2b",
    "summarize_execution",
    "extract_output_from_execution",
    "failed",

    # File/artifact management
    "list_tree",
    "print_tree",
    "download_all_as_tar",
    "download_folder_recursive",
    "FileEntry"
]