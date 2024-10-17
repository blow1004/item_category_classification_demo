from setuptools import setup, find_packages

setup(
    name='item_category_classification_demo',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        # 여기에 추가적인 패키지를 적어주세요
    ],
)
