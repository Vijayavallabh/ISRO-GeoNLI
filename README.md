# Team_46

This is the repository of Team_46

## Environment Setup

This project uses `uv` for managing Python virtual environments and dependencies. `uv` is a fast, drop-in replacement for pip and virtualenv, built in Rust.

### Prerequisites

- Python 3.10 or higher
- Git

### Installation

1. Install `uv`:
   - On macOS and Linux:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
   - On Windows:
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.sh | iex"
     ```
   - Or via pip (if you have Python installed):
     ```bash
     pip install uv
     ```

2. Clone the repository:
   ```bash
   git clone https://github.com/Vijayavallabh/Team_46.git
   cd Team_46
   ```

### Create and Activate Virtual Environment

1. Create a virtual environment using `uv`:
   ```bash
   uv venv
   ```
   This creates a `.venv` directory in the project root.

2. Activate the virtual environment:
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows (PowerShell):
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - On Windows (cmd):
     ```cmd
     .venv\Scripts\activate.bat
     ```

### Install Dependencies

- If the project has a `requirements.txt`:
  ```bash
  uv pip install -r requirements.txt
  ```
- If using a `pyproject.toml` (uv supports it):
  ```bash
  uv sync
  ```
- To add new dependencies:
  ```bash
  uv add package-name
  ```


Adjust the command based on your project's entry point.

### Deactivating the Environment

To deactivate the virtual environment:

```bash
deactivate
```

### Additional Notes

- `uv` automatically handles dependency resolution and caching for faster installs.
- For more information, visit the [uv documentation](https://docs.astral.sh/uv/).
- If you encounter issues, ensure Python is installed and `uv` is up to date.
