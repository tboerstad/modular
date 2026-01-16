# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

"""Entrypoint for generalized click cli."""

from __future__ import annotations

import logging

import click


class ConditionalLevelFormatter(logging.Formatter):
    """Formatter that only shows log level prefix for WARNING and above."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
    ):
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        # Build level prefix only for WARNING and above
        if record.levelno >= logging.WARNING:
            level_prefix = f"{record.levelname}: "
        else:
            level_prefix = ""

        # Format timestamp
        timestamp = self.formatTime(record, self.datefmt)
        msecs = f".{int(record.msecs):03d}"

        # Build the final message
        message = record.getMessage()
        formatted = f"{timestamp}{msecs} {level_prefix}{message}"

        # Handle exception info if present
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if formatted[-1:] != "\n":
                formatted = formatted + "\n"
            formatted = formatted + record.exc_text
        if record.stack_info:
            if formatted[-1:] != "\n":
                formatted = formatted + "\n"
            formatted = formatted + self.formatStack(record.stack_info)

        return formatted


class PrefixFormatter(logging.Formatter):
    """Formatter that adds a prefix to log messages."""

    def __init__(self, prefix: str, base_formatter: logging.Formatter):
        super().__init__()
        self.prefix = prefix
        self.base_formatter = base_formatter

    def format(self, record: logging.LogRecord) -> str:
        # Format the message using the base formatter
        formatted_message = self.base_formatter.format(record)
        # Add the prefix to the message
        return f"{self.prefix} {formatted_message}"


def configure_cli_logging(
    level: str = "INFO", log_prefix: str | None = None
) -> None:
    """Configure logging for CLI operations without using Settings.

    Args:
        level: Log level as string (DEBUG, INFO, WARNING, ERROR)
        log_prefix: Optional prefix to prepend to all log messages
    """
    if level == "INFO":
        log_level = logging.INFO
    elif level == "DEBUG":
        log_level = logging.DEBUG
    elif level == "WARNING":
        log_level = logging.WARNING
    elif level == "ERROR":
        log_level = logging.ERROR
    else:
        raise ValueError(f"Unsupported log level: {level}")

    # Clear existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Create console handler
    console_handler = logging.StreamHandler()
    console_formatter: logging.Formatter = ConditionalLevelFormatter(
        datefmt="%H:%M:%S",
    )

    # Apply log prefix if provided
    if log_prefix is not None:
        console_formatter = PrefixFormatter(log_prefix, console_formatter)

    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)

    # Set up log filtering for MAX components
    components_to_log = [
        "root",
        "max.entrypoints",
        "max.pipelines",
        "max.serve",
    ]

    def log_filter(record: logging.LogRecord) -> bool:
        """Filter to only show logs from MAX components."""
        return any(
            record.name == component or record.name.startswith(component + ".")
            for component in components_to_log
        )

    console_handler.addFilter(log_filter)

    # Configure root logger
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Reduce noise from external libraries
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("sse_starlette.sse").setLevel(
        max(log_level, logging.INFO)
    )


class ModelGroup(click.Group):
    def get_command(
        self, ctx: click.Context, cmd_name: str
    ) -> click.Command | None:
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        supported = ", ".join(self.list_commands(ctx))
        ctx.fail(
            f"Command not supported: {cmd_name}\n"
            f"Supported commands: {supported}"
        )
