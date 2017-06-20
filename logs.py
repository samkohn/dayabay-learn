import logging
import subprocess

logging.basicConfig(format='%(levelname)s:\t%(message)s')

def log_with_git_hash(text, logfile):
    commit_hash = subprocess.check_output(['git', 'describe',
        '--always', '--long']).strip().split('-')[-1][1:]
    tree_is_dirty = (len(subprocess.check_output(['git', 'diff-index', 'HEAD']))
            > 0)
    if tree_is_dirty:
        commit_hash += '+'
    runlogger = logging.getLogger('runlogger')
    runlogger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)-35s %(message)s',
            datefmt='%Y-%b-%d %H:%M, git ' + commit_hash + ']')
    handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    runlogger.addHandler(handler)
    runlogger.debug(text)
