from setuptools import setup
import os
from glob import glob

package_name = 'car_vroom_vroom'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Model-free machine learning for F1Tenth car',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ml_train = car_vroom_vroom.ml_train:main',
            'ml_inference = car_vroom_vroom.ml_inference:main',
            'safety_node = car_vroom_vroom.safety_node:main',
        ],
    },
)

