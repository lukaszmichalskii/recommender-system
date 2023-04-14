from platform import platform


def get_current_os() -> str:
    if is_linux_os():
        return "linux"
    return "windows"


def is_linux_os() -> bool:
    return platform().find("Linux") != -1


class Environment:
    """
    Class for storing user specific configuration overwritten using environmental variables
    """

    def __init__(self, env):
        self.os = get_current_os()

    @staticmethod
    def from_env(env):
        return Environment(env)

    def to_info_string(self):
        return "os: {} ".format(self.os)
