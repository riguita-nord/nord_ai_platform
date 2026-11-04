"""Offline tests for the one-command shell installation and launcher."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy(ROOT / 'install.sh', self.root / 'install.sh')
        shutil.copy(ROOT / 'nord', self.root / 'nord')
        # Simulate an already-installed environment; never download Torch in tests.
        venv = self.root / '.venv' / 'bin'
        venv.mkdir(parents=True)
        fake_python = venv / 'python'
        fake_python.write_text('''#!/usr/bin/env bash
if [[ "$1" == "-c" ]]; then exit 0; fi
if [[ "$1" == "-m" && "$2" == "nord_brain" ]]; then
  printf 'RUNNING profile=%s args=%s\\n' "$NORD_BRAIN_PRESET" "$*"
  exit 0
fi
printf 'Unexpected Python command: %s\\n' "$*" >&2
exit 55
''')
        fake_python.chmod(0o755)
        self.env = os.environ.copy()
        self.env['NORD_BRAIN_PRESET'] = 'tiny'

    def run_shell(self, *args):
        return subprocess.run(['bash', *args], cwd=self.root, env=self.env, text=True, capture_output=True)

    def test_scripts_parse(self):
        for name in ('install.sh', 'nord'):
            self.assertEqual(self.run_shell('-n', name).returncode, 0)

    def test_one_command_installs_and_starts(self):
        p = self.run_shell('install.sh')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('RUNNING profile=tiny', p.stdout)

    def test_install_only_does_not_start(self):
        p = self.run_shell('install.sh', '--install-only')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn('RUNNING', p.stdout)

    def test_launcher_passes_cli_arguments(self):
        p = self.run_shell('nord', '--data', './dados')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('args=-m nord_brain --data ./dados', p.stdout)

    def test_help_without_installation(self):
        p = self.run_shell('install.sh', '--help')
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('bash install.sh', p.stdout)


if __name__ == '__main__':
    unittest.main()
