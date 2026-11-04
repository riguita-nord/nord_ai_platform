"""Offline installer tests including Debian/Ubuntu ensurepip recovery.

No package manager calls or downloads occur: all commands are mocked.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.mockbin=self.root/'mockbin'
        self.mockbin.mkdir()
        shutil.copy(ROOT/'install.sh',self.root/'install.sh')
        shutil.copy(ROOT/'nord',self.root/'nord')
        (self.root/'nord').chmod(0o755)
        self.env=os.environ.copy()
        self.env['NORD_BRAIN_PRESET']='tiny'
        self.env['PATH']=str(self.mockbin)+os.pathsep+self.env['PATH']
        self.env['PYTHON']=str(self.mockbin/'python3')
        self.env['MOCK_APT_FLAG']=str(self.root/'apt-installed')
        self.env['MOCK_APT_LOG']=str(self.root/'apt.log')
        # Fake successful env python, used for all normal installer tests
        self.fake_venv=self.root/'.venv'/'bin'/'python'
        self.fake_venv.parent.mkdir(parents=True)
        self.fake_venv.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then exit 0; fi
if [[ "$1" == "-m" && "$2" == "pip" && "$3" == "--version" ]]; then exit 0; fi
if [[ "$1" == "-m" && "$2" == "nord_brain" ]]; then printf 'RUNNING args=%s\\n' "$*"; exit 0; fi
exit 55
''')
        self.fake_venv.chmod(0o755)
        (self.mockbin/'python3').write_text('''#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then
 case "$2" in *version_info*) echo 'python3.12-venv'; exit 0;; *) exit 0;; esac
fi
if [[ "$1" == "-m" && "$2" == "venv" ]]; then
 if [[ -f "$MOCK_APT_FLAG" ]]; then
  mkdir -p "$4/bin"
  cp "$(dirname "$0")/working-venv-python" "$4/bin/python"
  exit 0
 fi
 mkdir -p "$4/bin"
 cp "$(dirname "$0")/broken-venv-python" "$4/bin/python"
 echo 'The virtual environment was not created successfully because ensurepip is not available.' >&2
 echo 'apt install python3.12-venv' >&2
 exit 1
fi
exit 50
''')
        (self.mockbin/'python3').chmod(0o755)
        shutil.copy(self.fake_venv,self.mockbin/'working-venv-python')
        (self.mockbin/'broken-venv-python').write_text('#!/usr/bin/env bash\nexit 1\n')
        (self.mockbin/'broken-venv-python').chmod(0o755)
        (self.mockbin/'apt-get').write_text('''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$MOCK_APT_LOG"
touch "$MOCK_APT_FLAG"
''')
        (self.mockbin/'apt-get').chmod(0o755)
        (self.mockbin/'sudo').write_text('''#!/usr/bin/env bash
exec "$@"
''')
        (self.mockbin/'sudo').chmod(0o755)

    def call(self,*args):
        return subprocess.run(['bash',*args],cwd=self.root,env=self.env,capture_output=True,text=True)

    def test_bash_syntax(self):
        for name in ('install.sh','nord'):
            self.assertEqual(self.call('-n',name).returncode,0)

    def test_existing_venv_starts_directly(self):
        r=self.call('install.sh')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('RUNNING',r.stdout)
        self.assertFalse((self.root/'apt.log').exists())

    def test_install_only(self):
        r=self.call('install.sh','--install-only')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertNotIn('RUNNING',r.stdout)

    def test_launcher_passes_data(self):
        r=self.call('nord','--data','./dados')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('--data ./dados',r.stdout)

    def test_help(self):
        self.fake_venv.unlink()
        r=self.call('install.sh','--help')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('bash install.sh',r.stdout)

    def test_partial_venv_ensurepip_failure_installs_package_and_recovers(self):
        self.fake_venv.write_text('#!/usr/bin/env bash\nexit 1\n')
        self.fake_venv.chmod(0o755)
        r=self.call('install.sh')
        self.assertEqual(r.returncode,0,r.stdout+'\n'+r.stderr)
        self.assertIn('ensurepip',r.stdout+r.stderr)
        self.assertIn('RUNNING',r.stdout)
        self.assertIn('python3.12-venv',(self.root/'apt.log').read_text())
        self.assertTrue(self.fake_venv.exists())

    def test_launcher_recovers_partial_environment(self):
        self.fake_venv.write_text('#!/usr/bin/env bash\nexit 1\n')
        self.fake_venv.chmod(0o755)
        r=self.call('nord')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertIn('RUNNING',r.stdout)
        self.assertIn('python3.12-venv',(self.root/'apt.log').read_text())

    def test_failed_venv_creation_does_not_touch_brain_data(self):
        self.fake_venv.unlink()
        data=self.root/'memory.sqlite3'
        data.write_text('KEEP')
        r=self.call('install.sh','--install-only')
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(data.read_text(),'KEEP')

    def test_existing_torch_but_missing_numpy_is_repaired(self):
        flag=self.root/'numpy-installed'
        self.env['MOCK_NUMPY_FLAG']=str(flag)
        self.fake_venv.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then
  if [[ "$2" == *numpy* ]]; then
    [[ -f "$MOCK_NUMPY_FLAG" ]]
    exit $?
  fi
  exit 0
fi
if [[ "$1" == "-m" && "$2" == "pip" && "$3" == "--version" ]]; then exit 0; fi
if [[ "$1" == "-m" && "$2" == "pip" && "$3" == "install" ]]; then
  if [[ "$4" == "numpy>=1.24,<3" ]]; then touch "$MOCK_NUMPY_FLAG"; exit 0; fi
  exit 30
fi
if [[ "$1" == "-m" && "$2" == "nord_brain" ]]; then
  printf 'RUNNING args=%s\\n' "$*"; exit 0
fi
exit 55
''')
        self.fake_venv.chmod(0o755)
        result=self.call('install.sh')
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertTrue(flag.exists())
        self.assertIn('NumPy',result.stdout)
        self.assertIn('RUNNING',result.stdout)
        result=self.call('install.sh')
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertNotIn('A instalar NumPy',result.stdout)

if __name__=='__main__':unittest.main()
