# run it:
# .\ws_install_venv.ps1 -envName intro-rag

# Get the environment name from the command line
param (
    [string]$envName
)

# Check if environment name is provided
if (-not $envName) {
    Write-Output "Please provide an environment name using --env <name>"
    exit
}

# Define the path to the virtual environment based on the provided environment name
$envPath = "C:/Users/ecepeda/envs/$envName"

# Check if the virtual environment folder exists
if (Test-Path $envPath) {
    Write-Output "Virtual environment already exists. Skipping creation."
} else {
    # Create a Python virtual environment
    python -m venv $envPath
}

# Activate the virtual environment
& "$envPath/Scripts/Activate.ps1"

# Update pip
python -m pip install --upgrade pip

# Install requirements from a file
pip install -r requirements.txt

Write-Output "Virtual environment '$envName' has been set up."
