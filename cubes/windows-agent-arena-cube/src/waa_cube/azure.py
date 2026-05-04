"""WAA Azure resource configuration.

Provision the gallery image once before running evaluations:

    uv run recipes/waa/eval_azure_waa_kusha.py

This will provision Kusha's pre-built image from HuggingFace on first run,
then launch evaluations.

Usage::

    from waa_cube.azure import WAA_WINDOWS_RESOURCE
    from cube_infra_azure import AzureInfraConfig

    infra = AzureInfraConfig(resource_group=os.environ["AZURE_RESOURCE_GROUP"])
    bench = WAABenchmark(infra=infra, default_tool_config=ComputerConfig())
"""

from cube.resource import VMResourceConfig

WAA_WINDOWS_RESOURCE = VMResourceConfig(
    name="waa-windows-vm",
    source_url="https://huggingface.co/datasets/kushasareen/waa-windows-image/resolve/main/waa-windows-prepared.qcow2",
    default_ttl_seconds=60 * 60 * 2,
    min_cpu_cores=8,
    min_ram_gb=8,
    uefi=True,
    tpm=True,
    os_type="windows",
    specialized=True,
    # Chrome/MSEdge tasks attach Playwright over CDP at host:9222 (forwarded
    # via socat inside the VM to whichever port the browser was launched on).
    # VLC tasks expose 8080. Both need their own host-side tunnel.
    forwarded_ports=[9222, 8080],
)
