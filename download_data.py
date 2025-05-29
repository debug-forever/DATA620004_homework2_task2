import os
import wget
import tarfile
from pathlib import Path

def download_voc_dataset():
    # 创建数据目录
    data_dir = Path('./data/VOCdevkit')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # VOC2007数据集URLs
    voc2007_urls = [
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
        'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    ]
    
    # VOC2012数据集URL
    voc2012_url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    
    def download_and_extract(url, target_dir):
        print(f"Downloading {url}...")
        filename = wget.download(url)
        print(f"\nExtracting {filename}...")
        with tarfile.open(filename, 'r') as tar:
            tar.extractall(path=target_dir)
        os.remove(filename)
        print(f"Finished processing {filename}")
    
    # 下载并解压数据集
    try:
        for url in voc2007_urls:
            download_and_extract(url, './data')
        download_and_extract(voc2012_url, './data')
        print("Dataset download and extraction completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
if __name__ == '__main__':
    download_voc_dataset()