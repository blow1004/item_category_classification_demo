from setuptools import setup, find_packages

setup(
    name='item_category_classification_demo',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'fasttext',
        'tokenizers',
        'jinja2',
        'aiofiles',  # Static files 처리를 위한 패키지
        'python-multipart',  # 파일 업로드를 위한 패키지
        # 여기에 추가적인 패키지를 적어주세요
    ],
)
