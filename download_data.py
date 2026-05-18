# 파일명: download_data.py
import os
import shutil
import urllib.request
import zipfile
from tqdm import tqdm

# 글로벌 설정
URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
DATA_DIR = "data"
ZIP_PATH = os.path.join(DATA_DIR, "tiny-imagenet-200.zip")
EXTRACT_DIR = DATA_DIR

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Downloading Tiny-ImageNet") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 1. 데이터셋 다운로드
    if not os.path.exists(ZIP_PATH):
        print("[1/3] 스탠퍼드 서버에서 Tiny-ImageNet 다운로드를 시작합니다...")
        download_url(URL, ZIP_PATH)
        print("\n✅ 다운로드 완료!")
    else:
        print("[1/3] 이미 다운로드된 압축 파일이 존재합니다. 다운로드를 건너뜁니다.")
        
    # 2. 압축 해제 (리눅스 unzip 명령어 없이 파이썬 내장 기능 활용)
    TARGET_DIR = os.path.join(DATA_DIR, "tiny-imagenet-200")
    if not os.path.exists(TARGET_DIR):
        print("[2/3] 파이썬을 이용해 압축을 해제합니다. 잠시만 기다려주세요...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
        print("✅ 압축 해제 완료!")
    else:
        print("[2/3] 압축이 풀린 폴더가 이미 존재합니다. 압축 해제를 건너뜁니다.")
        
    # 3. Validation 디렉토리를 PyTorch ImageFolder 구조에 맞게 자동 정렬
    val_dir = os.path.join(TARGET_DIR, "val")
    val_images_dir = os.path.join(val_dir, "images")
    annotations_file = os.path.join(val_dir, "val_annotations.txt")
    formatted_dir = os.path.join(val_dir, "images_formatted")
    
    if os.path.exists(annotations_file):
        print("[3/3] PyTorch 데이터 로더 인식을 위해 Validation 폴더 구조를 재정렬합니다...")
        os.makedirs(formatted_dir, exist_ok=True)
        
        with open(annotations_file, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            tokens = line.strip().split()
            if not tokens:
                continue
            img_name = tokens[0]
            class_id = tokens[1]
            
            src_path = os.path.join(val_images_dir, img_name)
            dst_class_dir = os.path.join(formatted_dir, class_id)
            os.makedirs(dst_class_dir, exist_ok=True)
            dst_path = os.path.join(dst_class_dir, img_name)
            
            if os.path.exists(src_path):
                shutil.move(src_path, dst_path)
                
        # 임시 폴더 교체 및 텍스트 파일 정리
        shutil.rmtree(val_images_dir)
        os.rename(formatted_dir, val_images_dir)
        os.remove(annotations_file)
        
        # 다운로드 완료 후 용량 확보를 위해 원본 zip 파일 삭제
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
            
        print("\n🎉 모든 데이터셋 준비가 성공적으로 끝났습니다! 🎉")
    else:
        print("[3/3] Validation 폴더가 이미 정렬되어 있거나 설정 파일이 없습니다. 확인해 보세요.")

if __name__ == "__main__":
    main()
