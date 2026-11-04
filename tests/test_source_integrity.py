"""Catch syntax errors in all shipped Python modules before release.

The previous self-knowledge update accidentally duplicated a parser fragment,
causing import-time SyntaxError. Compiling source does not require torch.
"""
import pathlib
import unittest

ROOT=pathlib.Path(__file__).resolve().parents[1]

class SourceIntegrityTests(unittest.TestCase):
    def test_all_python_files_compile(self):
        paths=sorted((ROOT/'nord_brain').rglob('*.py'))+sorted((ROOT/'tests').glob('test_*.py'))
        self.assertGreater(len(paths),8)
        for path in paths:
            with self.subTest(path=str(path.relative_to(ROOT))):
                compile(path.read_bytes(),str(path),'exec')

if __name__ == '__main__':
    unittest.main()
