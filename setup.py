import os
import subprocess
import sys

def create_virtual_environment():
    """Creates a virtual environment named 'venv'."""
    if not os.path.exists('venv'):
        print("Creating virtual environment...")
        subprocess.check_call([sys.executable, '-m', 'venv', 'venv'])
        print("Virtual environment 'venv' created.")
    else:
        print("Virtual environment 'venv' already exists.")

def install_requirements():
    """Installs required packages from requirements.txt."""
    print("Installing required packages...")
    pip_executable = os.path.join('venv', 'bin', 'pip') if sys.platform != 'win32' else os.path.join('venv', 'Scripts', 'pip.exe')
    subprocess.check_call([pip_executable, 'install', '-r', 'requirements.txt'])
    print("All packages have been installed successfully.")

def main():
    """Main function to set up the environment."""
    create_virtual_environment()
    install_requirements()
    print("\nSetup complete! To activate the virtual environment, run:")
    if sys.platform == 'win32':
        print(r".\venv\Scripts\activate")
    else:
        print("source venv/bin/activate")
    print("\nThen, you can run the main application using:")
    print("python app.py")


if __name__ == "__main__":
    main()
