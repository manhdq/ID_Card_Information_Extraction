from ...general.utils.env import collect_env as collect_base_env
from ...general.utils.version_utils import get_git_hash
from ..version import __version__ as detection_version


def collect_env():
    """Collect the information of the running environments."""
    env_info = collect_base_env()
    env_info['MMDetection'] = detection_version + '+' + get_git_hash()[:7]
    return env_info


if __name__ == '__main__':
    for name, val in collect_env().items():
        print(f'{name}: {val}')