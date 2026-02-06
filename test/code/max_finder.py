import numpy as np
import glob
from PIL import Image
from tqdm import tqdm

def find_global_max(image_paths, sample_count=200):
    sampled_paths = np.random.choice(image_paths, sample_count, replace=False)
    max_val = 0.0
    for p in tqdm(sampled_paths, desc="이미지 처리 중", unit="img"):
        img = np.array(Image.open(p)).astype(np.float32)
        curr_max = img.max()
        if curr_max > max_val:
            max_val = curr_max
    return max_val

if __name__ == "__main__":
    scene_dir = "/home/hjahn/mnt/ssd1/data/hjahn/scene_and_label/image/0/"
    image_paths = glob.glob(f"{scene_dir}/*.png")
    max_brightness = find_global_max(image_paths)
    print(f"Global max brightness: {max_brightness}")

# 결과가 0.8이 나왔다면, 안전하게 1.0으로 잡는 식
# 만약 시뮬레이터 상에서 이론적 최대 밝기가 있다면 그 값을 쓰는 게 Best입니다.