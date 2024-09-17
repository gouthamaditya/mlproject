from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements
    from the given file path.
    """
    requirements = []
    # Using 'with' ensures the file is closed after reading
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Removing newline characters
        requirements = [req.strip() for req in requirements]

        # Removing '-e .' if present in the requirements
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Goutham',
    author_email='gouthamaditya.m@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
