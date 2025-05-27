from setuptools import find_packages, setup

package_name = 'rl_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools',
    'gymnasium',
    'stable-baselines3[extra]',
    'numpy',
    'rclpy',],
    zip_safe=True,
    maintainer='gunner',
    maintainer_email='ebyj23@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rl_controller = rl_control.wheel_RL_controller:main',
        ],
    },
)
