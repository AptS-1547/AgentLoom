from pathlib import Path
import shutil
import sys


source = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()

if target.exists():
    shutil.rmtree(target)

ignore = shutil.ignore_patterns(
    ".git",
    ".vs",
    ".agents",
    ".codex",
    ".cache",
    ".pytest_cache",
    "__pycache__",
    "build",
    "deps",
    "node_modules",
    "target",
    "vcpkg",
    "vcpkg_installed",
)
shutil.copytree(source, target, ignore=ignore)

required = (
    target / "CMakeLists.txt",
    target / "cmake" / "AgentLoomDependencies.cmake",
    target / "cmake" / "AgentLoomPackaging.cmake",
)
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit(f"prepared Linux source tree is incomplete: {', '.join(missing)}")
