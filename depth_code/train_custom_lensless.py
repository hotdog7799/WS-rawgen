import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

class LenslessDataset(Dataset):
    def __init__(self, data_dir):
        """
        data_dir: 렌더링된 이미지와 npz 파일이 있는 최상위 경로
        """
        self.data_dir = data_dir
        
        # 이미지 파일 목록 수집 (확장자가 다를 경우 수정 필요)
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        # npz 파일 목록 수집
        self.depth_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        # 데이터 쌍이 맞는지 치명적 오류 검증
        assert len(self.image_paths) == len(self.depth_paths), \
            f"데이터 개수 불일치! 이미지: {len(self.image_paths)}장, 깊이맵: {len(self.depth_paths)}개"
        
        assert len(self.image_paths) > 0, "데이터를 찾을 수 없습니다. 경로를 확인하세요."
        
        # 이미지를 PyTorch Tensor로 변환하기 위한 transform
        self.transform = transforms.Compose([
            transforms.ToTensor() # 0~255 값을 0.0~1.0으로 정규화 및 (C, H, W) 형태로 변환
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. 이미지 로드
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB') # 흑백이면 'L'로 변경
        img_tensor = self.transform(image)
        
        # 2. 깊이맵 npz 로드
        depth_path = self.depth_paths[idx]
        npz_data = np.load(depth_path)
        
        # npz 내부의 배열 이름을 알아야 합니다. 
        # 블렌더에서 저장할 때 쓰인 key(예: 'depth', 'arr_0' 등)를 정확히 입력해야 합니다.
        # 일반적인 기본값인 'arr_0'으로 임시 지정합니다.
        depth_array = npz_data['dmap'] 
        
        # (H, W) 형태의 numpy 배열을 (1, H, W) 형태의 텐서로 변환
        depth_tensor = torch.from_numpy(depth_array).unsqueeze(0).float()
        
        return img_tensor, depth_tensor

# --- 검증을 위한 실행 코드 ---
if __name__ == "__main__":
    # TODO: 실제 2만 장 데이터가 있는 경로로 수정할 것
    DATA_PATH = "/home/hjahn/mnt/nas/Research/HJA/rendering/20260324_133624_20000/image/" 
    
    dataset = LenslessDataset(data_dir=DATA_PATH)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 첫 번째 배치(Batch)만 뽑아서 형태(Shape) 확인
    imgs, depths = next(iter(dataloader))
    print(f"Total dataset size: {len(dataset)}")
    print(f"Image tensor shape: {imgs.shape} # 예상: [4, 3, H, W]")
    print(f"Depth tensor shape: {depths.shape} # 예상: [4, 1, H, W]")