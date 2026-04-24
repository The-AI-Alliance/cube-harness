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
)
