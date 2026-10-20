"""Plot actual fallback sampling coordinates and mask weights, without inference."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.nn import functional as F

from run import ROOT, LOCAL, old, geometry, sampled

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def main():
    case = pd.read_csv(ROOT / 'mse_aligned/case_review.csv')
    row = case[(case.obs_id == 9180) & (case.pol == 'vh')].iloc[0]
    snapshots = json.loads((LOCAL / 'view_snapshot.json').read_text(encoding='utf-8'))
    view = next(v for v in snapshots if v['obs_id'] == 9180)['images']['vh']
    raw = np.asarray(Image.open(row.path).convert('L'))
    pix, masks, place = old.crop_pad(raw, old.masks_for(view), 160)
    g, radii = geometry(view)
    assert g[3] == 0 and not row.vh_disabled
    m = F.avg_pool2d(torch.from_numpy(masks)[None], 4)
    angles = torch.tensor([[g[0], g[1]]], dtype=torch.float32)
    _, mass = sampled(torch.ones((1, 1, 40, 40)), m, angles, torch.tensor([radii], dtype=torch.float32))
    mass = mass[0].numpy()
    # Exact weights from describe(): center * min(two endpoint masses).
    wa = mass[..., 1, 0] * np.minimum(mass[..., 1, 1], mass[..., 1, 2])
    wc = mass[..., 1, 0] * np.minimum(mass[..., 1, 3], mass[..., 1, 4])
    selected = []
    yy, xx = np.indices((40, 40))
    for role in (0, 2):
        candidates = np.argwhere((wa[role] > .05) & (wc[role] > .05))
        assert len(candidates)
        anchor = np.array([19.5, 19.5]) if role == 0 else np.array([16., 16.])
        y, x = candidates[np.argmin(((candidates-anchor)**2).sum(1))]
        selected.append((int(y), int(x)))
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    fig.suptitle('真实案例 9180 · VH：缺少可靠雷达参考时，第二套采样改用图像水平/垂直方向', fontsize=15)
    axes[0, 0].imshow(raw, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[0, 0].set_title('① 原始 PNG：64 × 64\n没有缩放；该路图像仍然使用')
    axes[0, 1].imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
    axes[0, 1].contour(masks[0], [.5], colors=['#19d793'], linewidths=1)
    axes[0, 1].contour(masks.sum(0), [.5], colors=['#e6b84a'], linewidths=1)
    axes[0, 1].set_title('② 补零到 160 × 160\n绿：目标框；黄：原图有效范围')
    rgb = np.ones((40, 40, 3)) * .94
    colors = np.array([[.1,.75,.45], [1.,.55,.1], [.25,.5,.9]])
    for role in (2, 1, 0):
        a = m[0, role].numpy()[..., None]
        rgb = rgb * (1-a) + colors[role] * a
    axes[0, 2].imshow(rgb, interpolation='nearest')
    axes[0, 2].scatter(xx.ravel(), yy.ravel(), s=1, c='#333333', alpha=.25)
    for label, (y,x) in zip(('A', 'B'), selected):
        axes[0, 2].text(x,y,label,color='black',fontsize=12,weight='bold')
    axes[0, 2].set_title('③ 实际 40 × 40 特征网格\n绿：框内 T；蓝：其余 R；本例无其他框 O')
    details = []
    for ax, role, label, (y,x) in zip(axes[1,:2], (0,2), ('A：框内', 'B：框外'), selected):
        ax.imshow(pix, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
        ax.contour(masks[0], [.5], colors=['#19d793'], linewidths=1)
        # Map feature-grid bin centers to the native canvas for display only.
        # The stem has an overlapping receptive field, not 4x4 disjoint pixels.
        cx, cy = 4*x+1.5, 4*y+1.5
        for angle, color, style in ((g[0], '#22d9ef','-'), (0., '#ff78dc','--')):
            for theta, radius in ((angle, radii[0]), (angle+np.pi/2, radii[1])):
                dx, dy = 4*radius*np.cos(theta), 4*radius*np.sin(theta)
                ax.plot([cx-dx,cx,cx+dx], [cy-dy,cy,cy+dy], style+'o',color=color,ms=4,lw=1.5)
        ax.set_xlim(cx-13,cx+13); ax.set_ylim(cy+13,cy-13)
        ax.set_title(f'④ {label}的一个真实采样中心\n青：船轴参考；粉虚线：图像方向回退')
        details.append(dict(role=role, feature_center_xy=[x,y],
            fallback_five_mask_masses=mass[role,y,x,1].tolist(),
            along_weight=float(wa[role,y,x]), across_weight=float(wc[role,y,x])))
    im = axes[1,2].imshow(wa[0], vmin=0, vmax=1,cmap='viridis',interpolation='nearest')
    axes[1,2].set_title('⑤ 框内横向成对统计的实际权重\n跨出该区域会减权；补零区不当有效内容')
    fig.colorbar(im, ax=axes[1,2], shrink=.8)
    for ax in axes.flat:
        ax.set_xlabel('x（当前面板坐标）'); ax.set_ylabel('y')
    fig.text(.5,.018,'仅显示未旋转的一次读出；实际对 0° / 90° / 180° / 270° 四次读出取平均。图中采样点是特征位置投影，不是直接读取单个灰度像素。',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.045,1,.94))
    fig.savefig(ROOT / 'fallback_9180_explained.png',dpi=140)
    plt.close(fig)
    record = dict(obs_id=9180,pol='vh',ship_angle_deg=float(np.degrees(g[0])),radar_valid=False,
        fallback_angle_deg=0,feature_radii=radii,native_canvas_offset_px=(np.array(radii)*4).tolist(),
        placement=place,selected_centers=details,
        note='No model rerun. Grid-bin overlay only; overlapping stem receptive fields remain.')
    (ROOT/'fallback_9180_details.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(record,ensure_ascii=False))


if __name__ == '__main__':
    main()
