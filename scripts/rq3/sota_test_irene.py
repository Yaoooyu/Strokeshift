import sys,torch
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from models.irene_3d import IRENE3D
m=IRENE3D(26).cuda().eval()
with torch.no_grad():y=m(torch.randn(2,26,device='cuda'),torch.randn(2,1,128,128,128,device='cuda'))
assert y.shape==(2,) and torch.isfinite(y).all();print('IRENE_FORWARD_PASS')
