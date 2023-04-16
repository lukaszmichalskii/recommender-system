from platform import platform


def enum(**params):
    return type("Enum", (), params)


STEPS = enum(
    RECOMMEND="recommend",
    FIND="find",
    SIMILAR="similar"
)

STEPS_CHOICES = [STEPS.RECOMMEND, STEPS.FIND, STEPS.SIMILAR]
STANDARD_STEPS = [STEPS.RECOMMEND]

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

    @staticmethod
    def from_env(env):
        return Environment(env)

    def to_info_string(self):
        return "os: {}, recommendations limit: {}, precision: {}".format(
            self.os,
            self.recommendations_limit,
            self.precision
        )
