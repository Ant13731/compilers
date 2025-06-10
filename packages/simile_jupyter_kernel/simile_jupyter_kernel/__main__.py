from ipykernel.kernelapp import IPKernelApp
from . import SimileKernel

IPKernelApp.launch_instance(kernel_class=SimileKernel)
