"""Background script: sleeps a randomized interval, then opens the same
chat window with a gentle opener. Independent of app.py -- either can be
used alone.

Entry point -- must load .env itself (same reason as app.py).
"""
import argparse
import random
import time
import tkinter as tk

from dotenv import load_dotenv

load_dotenv()

from app import ChatApp

OPENER_MESSAGE = "Hey, how are you doing?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Randomized-interval nudge for Job Load Agent")
    parser.add_argument(
        "--interval-min",
        type=float,
        default=45,
        help="Minimum minutes between nudges (dev override, default 45).",
    )
    parser.add_argument(
        "--interval-max",
        type=float,
        default=90,
        help="Maximum minutes between nudges (dev override, default 90).",
    )
    args = parser.parse_args()

    while True:
        wait_minutes = random.uniform(args.interval_min, args.interval_max)
        time.sleep(wait_minutes * 60)

        root = tk.Tk()
        ChatApp(root, opener_message=OPENER_MESSAGE)
        root.mainloop()


if __name__ == "__main__":
    main()
