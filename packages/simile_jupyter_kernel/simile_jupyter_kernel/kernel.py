from typing import Any

from ipykernel.kernelbase import Kernel

# from simile.language import __version__ as simile_version
# from . import __version__

simile_version = "0.0.1"
__version__ = "0.0.1"


class SimileKernel(Kernel):
    implementation = "simile_kernel"
    implementation_version = __version__
    banner = f"Jupyter Simile Kernel, v{__version__} running Simile v{simile_version}"

    language_info = {
        "name": "simile",
        "version": simile_version,
        "mimetype": "text/x-simile",
        "file_extension": ".simile",
        "codemirror_mode": "simile",
        "pygments_lexer": "simile",
    }

    def do_execute(
        self,
        code,
        silent,
        store_history=True,
        user_expressions=None,
        allow_stdin=False,
    ):
        # Can't specify the types normally or mypy complains about the subclass
        # Jupyter's documentation specifies the types though
        assert isinstance(code, str), "Code must be a string"
        assert isinstance(silent, bool), "Silent must be a boolean"
        assert isinstance(store_history, bool), "Store history must be a boolean"
        assert isinstance(user_expressions, (dict, type(None))), "User expressions must be a dict or None"
        assert isinstance(allow_stdin, bool), "Stdin flag must be a boolean"

        if not silent:
            stream_content = {"name": "stdout", "text": f"Received:{code}"}
            self.send_response(self.iopub_socket, "stream", stream_content)

        return {
            "status": "ok",
            # The base class increments the execution count
            "execution_count": self.execution_count,
            "payload": [],
            "user_expressions": {},
        }
