import logging
import sys
import subprocess

def get_commit_hash():
    commit_hash = subprocess.check_output(['git', 'describe',
        '--always', '--long']).strip().split('-')[-1][1:]
    tree_is_dirty = (len(subprocess.check_output(['git', 'diff-index', 'HEAD']))
            > 0)
    if tree_is_dirty:
        commit_hash += '+'
    return commit_hash

def get_tee_logger(logfile):
    '''
    Gets a logger object which prints to sys.stdout and also to the
    specified logfile.

    '''
    commit_hash = get_commit_hash()
    teelogger = logging.getLogger('teelogger')
    teelogger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)-35s %(message)s',
            datefmt='%Y-%b-%d %H:%M, git ' + commit_hash + ']')
    filehandler = logging.FileHandler(logfile)
    consolehandler = logging.StreamHandler(sys.stdout)
    filehandler.setFormatter(formatter)
    consolehandler.setFormatter(formatter)
    teelogger.addHandler(filehandler)
    teelogger.addHandler(consolehandler)
    return teelogger

def log_with_git_hash(text, logfile):
    commit_hash = get_commit_hash()
    runlogger = logging.getLogger('runlogger')
    runlogger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)-35s %(message)s',
            datefmt='%Y-%b-%d %H:%M, git ' + commit_hash + ']')
    handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    runlogger.addHandler(handler)
    runlogger.debug(text)
