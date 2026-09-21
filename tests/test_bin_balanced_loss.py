import torch, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import train as T

mean, std = 0.0, 20.0
deg  = torch.tensor([[2., -2., 10., -10., 40., -40.]])
y    = (deg - mean) / std
v    = torch.ones_like(y)
err  = torch.tensor([[0., 0., 0., 0., 10., 10.]])
pred = ((deg + err - mean) / std).unsqueeze(-1)
mse = T.masked_mse(pred, y, v).item()
bal = T.masked_bin_balanced_mse(pred, y, v, mean, std).item()
print(f"  基本: mse={mse:.6f}  balanced={bal:.6f}  (期望 {0.25*2/6:.6f} / {0.25/3:.6f})")
assert abs(mse - 0.25*2/6) < 1e-6 and abs(bal - 0.25/3) < 1e-6

# 帧数不均才暴露差别: 直行 8 帧, 缓弯 2 帧, 急弯 2 帧
deg2 = torch.tensor([[2.,-2.,2.,-2.,2.,-2.,2.,-2., 10.,-10., 40.,-40.]])
y2 = (deg2 - mean)/std; v2 = torch.ones_like(y2)
e_s = torch.tensor([[10.]*8 + [0.]*4])
e_c = torch.tensor([[0.]*10 + [10.]*2])
f = lambda e: ((deg2 + e - mean)/std).unsqueeze(-1)
m_s, m_c = T.masked_mse(f(e_s),y2,v2).item(), T.masked_mse(f(e_c),y2,v2).item()
b_s, b_c = (T.masked_bin_balanced_mse(f(e_s),y2,v2,mean,std).item(),
            T.masked_bin_balanced_mse(f(e_c),y2,v2,mean,std).item())
print(f"  误差全在直行(8帧): mse={m_s:.6f}  balanced={b_s:.6f}")
print(f"  误差全在急弯(2帧): mse={m_c:.6f}  balanced={b_c:.6f}")
assert m_s > m_c, "mse 应更在意帧数多的一侧"
assert abs(b_s - b_c) < 1e-9, f"balanced 应一视同仁: {b_s} vs {b_c}"

# 必炸: 若 bin 边界取错(0/5 而非 5/15), 缓弯帧会被归进急弯桶, 下式将不再成立
only_gentle = torch.tensor([[0.]*8 + [10.,10.] + [0.,0.]])
b_g = T.masked_bin_balanced_mse(f(only_gentle),y2,v2,mean,std).item()
assert abs(b_g - b_c) < 1e-9, f"缓弯与急弯应各占 1/3 且相等: {b_g} vs {b_c}"
print(f"  误差全在缓弯(2帧): balanced={b_g:.6f}  -- 与急弯相等,证明三桶切分正确")
print("\n  ✓ mse 罚直行是急弯的 {:.1f} 倍(仅因帧多); bin-balanced 三桶完全等权".format(m_s/m_c))
