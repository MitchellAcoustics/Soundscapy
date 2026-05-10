#!/usr/bin/env python3
import re
import sys


def main():
    pyproject = sys.argv[1] if len(sys.argv) > 1 else "pyproject.toml"
    env_file = sys.argv[2] if len(sys.argv) > 2 else "environment.yml"

    # Extract version from pyproject.toml
    version = None
    with open(pyproject) as f:
        for line in f:
            if line.startswith("version = "):
                match = re.search(r'version = "([^"]+)"', line)
                if match:
                    version = match.group(1)
                    break

    if not version:
        print(f"Error: Could not extract version from {pyproject}", file=sys.stderr)
        sys.exit(1)

    print(f"Found soundscapy version: {version}")

    # Read and process environment.yml
    with open(env_file) as f:
        lines = f.readlines()

    output = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Skip name line
        if line.strip().startswith("name:"):
            i += 1
            continue

        # Replace pip editable installs
        if line.strip() == "- pip:":
            output.append(line)
            # Skip next 3 lines (the editable installs)
            i += 4
            # Add the new pip install
            output.append(f"  - soundscapy[r,audio]>={version}\n")
        else:
            output.append(line)
            i += 1

    # Write back
    with open(env_file, "w") as f:
        f.writelines(output)

    print(f"✓ Updated {env_file} with soundscapy version {version}")


if __name__ == "__main__":
    main()
