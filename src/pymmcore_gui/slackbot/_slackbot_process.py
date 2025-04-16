from __future__ import annotations

import base64
import io
import json
import logging
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from qtpy.QtCore import QProcess, Signal, Slot

logging.basicConfig(
    filename=Path(__file__).parent / "slackbot.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


ROBOT = ":robot:"
ALARM = ":rotating_light:"
MICROSCOPE = ":microscope:"


class SlackBotProcess(QProcess):
    """Process to run the SlackBot."""

    messageReceived = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.readyReadStandardOutput.connect(self.handle_message)
        self.readyReadStandardError.connect(self.handle_error)

    def stop(self) -> None:
        """Stop the SlackBot process."""
        self.kill()
        self.waitForFinished()

    def start(self) -> None:
        """Start the SlackBot in a new process.

        The process is started with the 'python' interpreter and the path to the
        '_slackbot.py' script (which contains the SlackBot class).
        """
        target = Path(__file__).parent / "_slackbot.py"
        super().start(sys.executable, [str(target)])
        if not self.waitForStarted():  # Check if the process started correctly
            msg = f"SlackBotProcess -> {ALARM} Failed to start SlackBotProcess! {ALARM}"
            logging.error(msg)
            warnings.warn(msg, stacklevel=2)
        else:
            logging.info(f"SlackBotProcess -> {ROBOT} SlackBotProcess started! {ROBOT}")

        self.send_message(
            {
                "icon_emoji": MICROSCOPE,
                "text": "Hello from Eve, the MicroManager's SlackBot!\n"
                "- `/run` -> Start the MDA Sequence\n"
                "- `/cancel` -> Cancel the current MDA Sequence\n"
                "- `/progress` -> Get the current MDA Sequence progress\n"
                "- `/image` -> Get the current MDA Sequence frame image\n"
                "- `/clear` -> Clear the chat from the SlackBot messages\n"
                "- `/mda` -> Get the current MDASequence",
            }
        )

    def _numpy_to_temp_png(self, img: np.ndarray) -> str:
        """Convert a NumPy array to a temporary PNG file."""
        if img.dtype != np.uint8:
            img_min = img.min()
            img_max = img.max()
            scaled_max = img_max * 0.8  # 80% of actual max
            img = np.clip(img, img_min, scaled_max)
            img = (255 * (img - img_min) / (scaled_max - img_min)).astype(np.uint8)
        image = Image.fromarray(img)

        # Create a temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image.save(temp_file.name)
        return temp_file.name

    def send_message(self, message: str | dict[str, Any] | np.ndarray) -> None:
        """Send a message to the process.

        The message is written to the process's stdin so that it can be read by the
        process and sent to the Slack channel.
        """
        msg = "numpy array image" if isinstance(message, np.ndarray) else message
        logging.info(f"SlackBotProcess -> received {msg}")

        if isinstance(message, dict):
            text = message.get("text", "")
            emoji = message.get("icon_emoji", "")
            message = json.dumps({"icon_emoji": emoji, "text": text})

        elif isinstance(message, np.ndarray):
            # serialize NumPy array to base64 string
            buffer = io.BytesIO()
            np.save(buffer, message)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            msg_dict = {
                "type": "image_array",
                "data": encoded,
                "text": "Here is the latest image from the MDA sequence.",
            }
            message = json.dumps(msg_dict)

        # send message to the process with a newline
        self.write((message + "\n").encode())
        # ensure the bytes are written
        if not self.waitForBytesWritten(1000):
            logging.error(f"SlackBotProcess -> Failed to write '{msg}' to the process!")
        else:
            logging.info(f"SlackBotProcess -> sent: '{msg}'")

    @Slot()  # type: ignore [misc]
    def handle_message(self) -> None:
        """Handle the message sent by the SlackBot in the new process process.

        This method is called when the process sends a message to stdout. Once received,
        the message is emitted as a signal to be connected to a slot in MicroManagerGUI.
        """
        message = self.readAllStandardOutput().data().decode()
        logging.info(f"SlackBotProcess -> received: {message}")
        self.messageReceived.emit(message)

    @Slot()  # type: ignore [misc]
    def handle_error(self) -> None:
        """Handle the error sent by the SlackBot in the new process process.

        This method is called when the process sends an error to stderr.
        """
        error = self.readAllStandardError().data().decode()
        logging.error(f"SlackBotProcess -> error received: {error}")
