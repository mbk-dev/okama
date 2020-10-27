from setuptools import setup

setup(
    name='okama',
    version='0.1',
    packages=['okama', 'tests'],
    package_data={'tests': ['*.csv']},
    install_requires=['pytest',
                      'pandas',
                      'requests',
                      'requests_cache',
                      'numpy',
                      'scipy',
                      'psycopg2',
                      'urllib3',
                      'matplotlib',
                      'eurostat',
                      'progress'],
    url='https://okama.io/',
    license='MIT',
    author='Sergey Kikevich',
    author_email='sergey@rostsber.ru',
    description='Modern Portfolio Theory (MPT) package',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7'
)
