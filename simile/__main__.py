import sys
import tempfile
import shutil
import json
import argparse

from ipykernel.kernelapp import IPKernelApp
from jupyter_client import kernelspec

from simile.jupyter_.kernel import SimileKernel


def install_kernel() -> None:
    # Install the kernel spec
    kernel_spec = {
        "argv": [
            "python",
            "-m",
            "simile",
            "-f",
            "{connection_file}",
        ],
        "display_name": "Simile Kernel",
        "language": "simile",
        "name": "simile_kernel",
    }
    temp_dir = tempfile.mkdtemp()
    try:
        # Write the kernel spec to a temporary directory
        json.dump(
            kernel_spec,
            open(f"{temp_dir}/kernel.json", "w"),
            indent=4,
        )
        kernelspec.KernelSpecManager().install_kernel_spec(
            temp_dir,
            kernel_name="simile_kernel",
            user=False,
            prefix=sys.prefix,
        )
    finally:
        # Clean up the temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Simile Jupyter Kernel")
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the Simile kernel spec",
    )

    args, unknown = parser.parse_known_args()
    if args.install:
        install_kernel()
        print("Simile kernel spec installed successfully.")
        return

    IPKernelApp.launch_instance(sys.argv, kernel_class=SimileKernel)


if __name__ == "__main__":
    main()
