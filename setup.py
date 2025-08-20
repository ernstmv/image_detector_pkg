from glob import glob
from setuptools import find_packages, setup

package_name = 'image_detector_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(
        exclude=[
            'test'
            ]
        ),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
        'cv_bridge',
        'numpy',
        'torch',
        ],
    zip_safe=True,
    maintainer='ernstmv',
    maintainer_email='ernestoroque777@gmail.com',
    description='The implementation of a 2d detector using yolo',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_detector_node = image_detector_pkg.image_detector_node:main'
        ],
    },
)
