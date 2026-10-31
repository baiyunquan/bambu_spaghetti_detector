"""The optimized backend must not transitively load the expensive runtime."""
import subprocess
import sys
import unittest
from pathlib import Path


class TestNativeImport(unittest.TestCase):
    def test_native_modules_do_not_import_torch(self):
        root = Path(__file__).resolve().parent.parent
        code = ("import sys;sys.path.insert(0,{!r});"
                "from native_cuda.runtime import DeviceTensor;"
                "assert not any(x=='torch' or x.startswith('torch.') or "
                "x=='torchvision' or x.startswith('torchvision.') for x in sys.modules);"
                "t=DeviceTensor(123,(1,3,2,2),'float32',object());"
                "assert t.nbytes==48").format(str(root / "inference"))
        result = subprocess.run([sys.executable, "-c", code], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode())


if __name__ == "__main__":
    unittest.main()
