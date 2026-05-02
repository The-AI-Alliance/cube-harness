import logging
import os
from typing import Union

logger = logging.getLogger("waa_cube.vm_backend.getters.info")


def get_vm_screen_size(env, config: dict) -> dict:
    return env.controller.get_vm_screen_size()


def get_vm_window_size(env, config: dict) -> dict:
    return env.controller.get_vm_window_size(app_class_name=config["app_class_name"])


def get_vm_wallpaper(env, config: dict) -> Union[str, bytes]:
    """Fetch the VM's current wallpaper to cache_dir. Raises ``FileNotFoundError``
    when the controller can't fetch one — distinguishing real infra failures
    from a wallpaper that the agent legitimately set to a 0-byte image."""
    _path = os.path.join(env.cache_dir, config["dest"])
    content = env.controller.get_vm_wallpaper()

    if content is None:
        raise FileNotFoundError("Controller returned None for wallpaper fetch")
    if not isinstance(content, bytes):
        raise TypeError(f"Wallpaper content must be bytes; got {type(content).__name__}")

    with open(_path, "wb") as f:
        f.write(content)
    return _path


def get_list_directory(env, config: dict) -> dict:
    return env.controller.get_vm_directory_tree(config["path"])
