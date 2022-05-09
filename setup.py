from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = [k for k in f.readlines()]

setup(
    name='recsys',
    version='0.1',
    description='Recommendation system',
    long_description='',
    long_description_content_type="text/markdown",
    author='Analytic Factory',
    author_email='analytic.factory@pedidosya.com',
    packages=find_packages(),
    install_requires=requirements,
)