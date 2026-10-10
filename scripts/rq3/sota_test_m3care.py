import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.m3care_3d import M3Care3D
m=M3Care3D(26).cuda().eval()
with torch.no_grad():
 y,l=m(torch.randn(2,26,device='cuda'),torch.randn(2,1,64,64,64,device='cuda'),torch.tensor([[1,1],[1,0]],device='cuda'))
assert y.shape==(2,) and torch.isfinite(l)
print('M3CARE_FORWARD_PASS')
