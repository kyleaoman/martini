import os
from _gensetup import gensetup
from twine.commands.upload import upload as twine_upload
from twine.settings import Settings as TwineSettings
from requests.exceptions import HTTPError
from getpass import getpass
import sys
import subprocess
from time import sleep

pkgname = 'astromartini'
condash = '/opt/local/anaconda/anaconda3-2018.12/etc/profile.d/conda.sh'


def run_chk(s, allow_fail=False):
    print('system> ', s)
    ec = os.system(s)
    if not allow_fail and (ec != 0):
        raise RuntimeError(ec)
    return


pyversion = sys.version.split(' ')[0]
pkgdir = os.path.dirname(os.path.realpath(__file__))
os.chdir(pkgdir)
version_file = os.path.join(pkgdir, 'martini', 'VERSION')
with open(version_file) as vf:
    version = tuple(vf.read().strip().split('.'))

print('Check that git master branch is ready and committed.')
passwd = getpass('Preparing {:s}-{:s}.{:s}, enter PyPI password to'
                 ' continue: '.format(pkgname, *version))

run_chk('pip install --upgrade setuptools wheel twine')

# new branch, or get in sync
run_chk('git checkout master')
try:
    run_chk('git branch {:s}.{:s}'.format(*version))
except RuntimeError as e:
    if e.args[0] != 32768:
        raise
    # branch exists, make sure it's up to date
    run_chk('git checkout {:s}.{:s}'.format(*version))
    run_chk('git merge master')
else:
    run_chk('git push --set-upstream origin {:s}.{:s}'.format(*version))

# rebuild docs
run_chk('git checkout docs')
run_chk('git merge {:s}.{:s}'.format(*version))
run_chk('rm -r {:s}'.format(os.path.join(pkgdir, 'docs', 'build')))
os.chdir(os.path.join(pkgdir, 'docs'))
run_chk('make html')
os.chdir(pkgdir)
run_chk('git add docs')
try:
    run_chk('git commit -m "Rebuilt docs."')
except RuntimeError as e:
    if e.args[0] == 256:  # nothing to commit
        pass
run_chk('git checkout master')
run_chk('git merge docs')
run_chk('git checkout {:s}.{:s}'.format(*version))
run_chk('git merge docs')


def distprefix(rc=None):
    if rc is None:
        return '{:s}-{:s}.{:s}'.format(pkgname, *version)
    else:
        return '{:s}-{:s}.{:s}.{:d}'.format(pkgname, *version, rc)


def whlname(rc=None):
    return distprefix(rc=rc) + '-py3-none-any.whl'


def tarname(rc=None):
    return distprefix(rc=rc) + '.tar.gz'


rcc = 0

errtxt = subprocess.run(
    'pip install --index-url https://test.pypi.org/simple/ --no-deps'
    ' {:s}==NULL'.format(pkgname),
    shell=True,
    universal_newlines=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
).stderr
compatible_test_versions = [
    v for v in errtxt.split('from versions: ')[1].split(')')[0].split(', ')
    if v.startswith('.'.join(version))
]
if len(compatible_test_versions) == 0:
    rcc = 0
else:
    previous = compatible_test_versions[-1]
    rcc = int(previous.split('.')[-1]) + 1

# generate setup.py for test-pypi
gensetup(for_pypi=True, test_subversion=rcc)
try:
    # generate distribution archives
    run_chk('python setup.py sdist bdist_wheel')
finally:
    # revert setup.py no matter what
    gensetup(for_pypi=False)
# upload to test.pypi
twine_settings = TwineSettings(
    username='kyleaoman',
    password=passwd,
    repository_url='https://test.pypi.org/legacy/'
)
try:
    twine_upload(
        twine_settings,
        (os.path.join('dist', '{:s}*'.format(distprefix(rc=rcc))), )
    )
finally:
    run_chk('rm {:s}.*'.format(os.path.join('dist', distprefix())))

# check that the uploaded package is visible on test pypi
errtxt = subprocess.run(
    'pip install --index-url https://test.pypi.org/simple/ --no-deps'
    ' {:s}==NULL'.format(pkgname),
    shell=True,
    universal_newlines=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
).stderr
while True:
    if '{:s}.{:s}.{:d}'.format(*version, rcc) in \
       errtxt.split('from versions: ')[1].split(')')[0].split(', '):
        break
    print('Test pypi index slow to sync, waiting 10s...')
    sleep(10)

# check that the uploaded package works
run_chk('conda create -y --name={:s}-test python={:s}'.format(
    pkgname, pyversion))
try:
    CP = subprocess.run(
        ". {:s} && ".format(condash) +
        "conda activate {:s}-test && ".format(pkgname) +
        "cd && " +
        "mkdir {:s}-test-scratch && ".format(pkgname) +
        "cd {:s}-test-scratch && ".format(pkgname) +
        "pip install numpy astropy scipy && " +
        "pip install --index-url https://test.pypi.org/simple/ --no-deps"
        " {:s}=={:s}.{:s}.{:d} && ".format(pkgname, *version, rcc) +
        "python -c 'from martini import demo;demo()'",
        shell=True,
        executable='/bin/bash'
    )
    if CP.returncode != 0:
        raise RuntimeError
    # should run additional tests in above subprocess
    # e.g. use actual python tests interface
finally:
    # this directory may not exist, allow failure
    run_chk('rm -r $HOME/{:s}-test-scratch'.format(pkgname), allow_fail=True)
    run_chk('conda env remove -y --name {:s}-test'.format(pkgname))

# tests passed, so let's prepare final upload
# generate setup.py for pypi
gensetup(for_pypi=True)
try:
    # generate distribution archives
    run_chk('python setup.py sdist bdist_wheel')
finally:
    # revert setup.py no matter what
    gensetup(for_pypi=False)

twine_settings = TwineSettings(
    username='kyleaoman',
    password=passwd
)
conf = input('Ready to upload {:s}-{:s}.{:s} to PyPI. Confirm?'.format(
    pkgname, *version))
if conf in ('y', 'Y', 'yes', 'YES', 'Yes'):
    twine_upload(
        twine_settings,
        (os.path.join('dist', '{:s}*'.format(distprefix())), )
    )
# still on version branch
run_chk('git add .')
run_chk('git commit -m "Ensure git branch matches pypi."')
run_chk('git push')
run_chk('git checkout master')
