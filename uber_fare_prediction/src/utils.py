"""
utils.py

Common helper functions used across the project.
"""

from datetime import datetime


def current_timestamp():
    """
    Return current timestamp.
    """

    return datetime.now()


def print_section(title):
    """
    Print formatted section header.
    """

    print("\n" + "=" * 60)

    print(title.upper())

    print("=" * 60)


def print_success(message):
    """
    Print success message.
    """

    print(f"SUCCESS: {message}")


def print_warning(message):
    """
    Print warning message.
    """

    print(f"WARNING: {message}")