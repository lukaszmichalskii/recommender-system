import multiprocessing
from platform import platform


def enum(**params):
    return type("Enum", (), params)


STEPS = enum(RECOMMEND="recommend", ANALYSIS="analysis", FIND="find", SIMILAR="similar")

STEPS_CHOICES = [STEPS.RECOMMEND, STEPS.ANALYSIS, STEPS.FIND, STEPS.SIMILAR]
STANDARD_STEPS = [STEPS.RECOMMEND, STEPS.ANALYSIS, STEPS.FIND]

SUPPORTED_FORMAT = ".csv"


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
        self.recommendations_limit = int(env.get("RECOMMENDATIONS_LIMIT", 10))
        self.precision = int(env.get("PRECISION", 200))
        self.api_key = env.get("API_KEY")
        self.cpu = int(env.get("CPU_THREADS", multiprocessing.cpu_count()))

    @staticmethod
    def from_env(env):
        return Environment(env)

    def to_info_string(self):
        return "OS: {}, threads: {}, recommendations limit: {}, precision: {}".format(
            self.os, self.cpu, self.recommendations_limit, self.precision
        )
