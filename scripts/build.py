#!/usr/bin/env python3

from pathlib import Path
import subprocess

# Define the project root directory
project_root_dir = Path(__file__).parent.parent.resolve()

# Define the build directory
build_dir = project_root_dir / 'build'

# Create the build directory if it doesn't exist
build_dir.mkdir(exist_ok=True)

# Run the CMake command to prepare the build
subprocess.check_call(['cmake', '-DCMAKE_BUILD_TYPE=Debug', '-G', 'Ninja', str(project_root_dir)], cwd=str(build_dir))

# Run the Ninja command to build the project
subprocess.check_call(['ninja'], cwd=str(build_dir))

