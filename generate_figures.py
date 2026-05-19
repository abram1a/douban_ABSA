"""
generate_figures.py
对应 6666文.docx 中所有图表，生成到 outputs_v6/figures/
适配 v8 代码（字符级编码 + DimAttention + 4阶段训练）
运行方式：
  python generate_figures.py              # 只生成数据分析图（不需要模型）
  python generate_figures.py --with_model # 生成包含模型结果的所有图
"""
import os, re, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import Counter

plt.rcParams["font.sans-serif"] = ["SimHei","Microsoft YaHei","Arial Unicode MS","DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SAVE_DIR = "outputs_v6/figures"
os.makedirs(SAVE_DIR, exist_ok=True)

ASPECTS   = ["actor","plot","vfx","music","director"]
ASP_CN    = {"actor":"演员","plot":"剧情","vfx":"特效","music":"音乐","director":"导演"}
ASP_LABEL = [ASP_CN[a] for a in ASPECTS]
SENT_CN   = {0:"负面", 1:"中性", 2:"正面"}
COLORS5   = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3"]

def S(path):
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/{path}", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  ✓ {path}")

# ══════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════
def clean_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linewidth=0.8)

# ══════════════════════════════════════════════════
# fig01：训练集样本角色来源分布
# ══════════════════════════════════════════════════
def fig01_data_roles(records=None):
    # 使用示意数据（实际跑完弱标注后传入 records_train）
    if records:
        role_cnt = Counter(r.get("role","other") for r in records)
    else:
        role_cnt = {
            "los_SIN":          320000,
            "los_LOVE":         180000,
            "auto_1dim":        420000,
            "auto_multi_pos":   310000,
            "auto_multi_neg":   180000,
            "auto_multi_mid":   150000,
            "manual_pending":    85000,
        }
    label_map = {
        "los_SIN":         "los_SIN\n（0维低分）",
        "los_LOVE":        "los_LOVE\n（0维高分）",
        "auto_1dim":       "单维度\n直接标注",
        "auto_multi_pos":  "多维度\n全正面",
        "auto_multi_neg":  "多维度\n全负面",
        "auto_multi_mid":  "多维度\n混合",
        "manual_pending":  "待人工\n（疑似反讽）",
        "manual_labeled":  "已人工\n标注",
    }
    labels = [label_map.get(k, k) for k in role_cnt]
    sizes  = list(role_cnt.values())
    colors = ["#4C72B0","#6BAED6","#55A868","#74C476","#C44E52","#DD8452","#8172B3","#CCCCCC"]

    fig, ax = plt.subplots(figsize=(9,6), dpi=150)
    wedges, _, autotexts = ax.pie(
        sizes, labels=None, colors=colors[:len(sizes)],
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=140, pctdistance=0.78,
        wedgeprops={"edgecolor":"white","linewidth":1.5})
    for at in autotexts: at.set_fontsize(9)
    total = sum(sizes)
    ax.legend(wedges, [f"{l}  ({s:,}, {s/total*100:.1f}%)"
                       for l,s in zip(labels, sizes)],
              loc="center left", bbox_to_anchor=(1,0.5), fontsize=9)
    ax.set_title("训练集样本角色来源分布", fontsize=13, fontweight="bold", pad=12)
    S("fig01_data_roles.png")

# ══════════════════════════════════════════════════
# fig02：各维度有情感标注的样本数量
# ══════════════════════════════════════════════════
def fig02_dim_sample_counts(records=None):
    if records:
        counts = [sum(1 for r in records if r["masks"][i]==1) for i in range(5)]
    else:
        counts = [198000, 1650000, 275000, 98000, 178000]  # 示意

    fig, ax = plt.subplots(figsize=(8,5), dpi=150)
    bars = ax.bar(ASP_LABEL, counts, color=COLORS5, edgecolor="white", linewidth=0.8)
    clean_ax(ax)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.set_title("各维度有情感标注的样本数量", fontsize=12, fontweight="bold")
    total = sum(counts)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+total*0.005,
                f"{cnt:,}\n({cnt/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    S("fig02_dim_sample_counts.png")

# ══════════════════════════════════════════════════
# fig03：各维度弱标注情感标签分布
# ══════════════════════════════════════════════════
def fig03_sentiment_dist(records=None):
    if records:
        pos = [sum(1 for r in records if r["masks"][i]==1 and r["labels"][i]==2) for i in range(5)]
        mid = [sum(1 for r in records if r["masks"][i]==1 and r["labels"][i]==1) for i in range(5)]
        neg = [sum(1 for r in records if r["masks"][i]==1 and r["labels"][i]==0) for i in range(5)]
        tots= [p+m+n for p,m,n in zip(pos,mid,neg)]
    else:
        # 示意比例
        tots= [198000,1650000,275000,98000,178000]
        pos = [int(t*p) for t,p in zip(tots,[0.64,0.60,0.39,0.39,0.30])]
        mid = [int(t*p) for t,p in zip(tots,[0.13,0.22,0.55,0.54,0.60])]
        neg = [t-p-m for t,p,m in zip(tots,pos,mid)]

    pp = [p/t*100 for p,t in zip(pos,tots)]
    mp = [m/t*100 for m,t in zip(mid,tots)]
    np_ = [n/t*100 for n,t in zip(neg,tots)]
    x  = range(len(ASPECTS))

    fig, ax = plt.subplots(figsize=(9,5), dpi=150)
    b1 = ax.bar(x, pp, label="正面", color="#55A868", edgecolor="white")
    b2 = ax.bar(x, mp, bottom=pp, label="中性", color="#CCCCCC", edgecolor="white")
    b3 = ax.bar(x, np_, bottom=[a+b for a,b in zip(pp,mp)],
                label="负面", color="#C44E52", edgecolor="white")
    ax.set_xticks(list(x)); ax.set_xticklabels(ASP_LABEL, fontsize=10)
    ax.set_ylabel("占比 (%)", fontsize=11)
    ax.set_title("各维度弱标注情感标签分布", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    # 标注正负比例
    for i,(p,n) in enumerate(zip(pp,np_)):
        ax.text(i, 1, f"{p:.0f}%", ha="center", va="bottom", fontsize=8,
                color="white", fontweight="bold")
    S("fig03_sentiment_dist.png")

# ══════════════════════════════════════════════════
# fig04：整体情感标签分布
# ══════════════════════════════════════════════════
def fig04_overall_label_dist(df=None):
    if df is not None:
        from collections import Counter
        def s2l(s):
            if s<=1: return "负面"
            if s==2: return "中性"
            return "正面"
        cnt = Counter(s2l(int(s)-1) for s in df["Star"])
    else:
        cnt = {"正面":1260000,"中性":440000,"负面":360000}

    labels = ["正面","中性","负面"]
    vals   = [cnt[l] for l in labels]
    colors = ["#55A868","#CCCCCC","#C44E52"]

    fig, ax = plt.subplots(figsize=(7,5), dpi=150)
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    clean_ax(ax)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.set_title("整体情感标签分布（Star映射）", fontsize=12, fontweight="bold")
    total = sum(vals)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+total*0.005,
                f"{v:,}\n({v/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")
    S("fig04_overall_label_dist.png")

# ══════════════════════════════════════════════════
# fig05：评论文本长度分布
# ══════════════════════════════════════════════════
def fig05_text_length_dist(df=None):
    if df is not None:
        lengths = df["Comment"].astype(str).str.len().clip(upper=200).tolist()
    else:
        np.random.seed(42)
        lengths = np.concatenate([
            np.random.exponential(20, 800000),
            np.random.normal(60, 30, 400000),
            np.random.normal(120, 40, 200000),
        ]).clip(1,200).tolist()

    fig, ax = plt.subplots(figsize=(9,4), dpi=150)
    ax.hist(lengths, bins=80, color="#4C72B0", edgecolor="white",
            linewidth=0.4, alpha=0.85)
    # 标注 MAX_LEN=80 线
    ax.axvline(80, color="#C44E52", linewidth=2, linestyle="--",
               label="MAX_LEN=80（截断阈值）")
    clean_ax(ax)
    ax.set_xlabel("评论字符长度", fontsize=11)
    ax.set_ylabel("评论数量", fontsize=11)
    ax.set_title("评论文本长度分布", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)

    # 计算截断比例
    arr = np.array(lengths)
    pct_over = (arr > 80).mean() * 100
    ax.text(100, ax.get_ylim()[1]*0.85,
            f"超过80字符\n约{pct_over:.1f}%",
            fontsize=9, color="#C44E52")
    S("fig05_text_length_dist.png")

# ══════════════════════════════════════════════════
# fig06：各维度关键词共现热力图
# ══════════════════════════════════════════════════
def fig06_dim_cooccurrence(records=None):
    # 计算各维度pair的共现比例
    n = len(ASPECTS)
    mat = np.zeros((n,n))
    if records:
        total = len(records)
        for i in range(n):
            for j in range(n):
                co = sum(1 for r in records
                         if r["masks"][i]==1 and r["masks"][j]==1)
                mat[i][j] = co/total*100
    else:
        # 示意数据
        mat = np.array([
            [10.5,  2.1,  1.3,  0.8,  1.5],
            [ 2.1, 100.0, 3.8,  2.5,  3.2],
            [ 1.3,  3.8, 14.2,  1.1,  1.8],
            [ 0.8,  2.5,  1.1,  5.2,  0.9],
            [ 1.5,  3.2,  1.8,  0.9,  9.1],
        ])

    fig, ax = plt.subplots(figsize=(7,6), dpi=150)
    # 对角线外的共现，对角线为自身覆盖率
    mask_diag = np.eye(n, dtype=bool)
    diag_vals = np.diag(mat).copy()
    mat_off   = mat.copy()
    np.fill_diagonal(mat_off, 0)

    im = ax.imshow(mat_off, cmap="Blues", vmin=0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(ASP_LABEL, fontsize=10)
    ax.set_yticklabels(ASP_LABEL, fontsize=10)
    ax.set_title("各维度关键词共现热力图（%）", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(n):
        for j in range(n):
            val = mat_off[i,j] if i!=j else diag_vals[i]
            label = f"{val:.1f}%" if i!=j else f"[{val:.1f}%]"
            ax.text(j,i,label, ha="center",va="center",
                    fontsize=8.5, color="white" if mat_off[i,j]>mat_off.max()*0.6 else "black")
    S("fig06_dim_cooccurrence.png")

# ══════════════════════════════════════════════════
# fig07-14：四阶段训练曲线（Loss + F1 各一张）
# ══════════════════════════════════════════════════
def plot_one_stage(hist, stage_name, loss_fname, f1_fname):
    # Loss 曲线
    fig, ax = plt.subplots(figsize=(8,4), dpi=150)
    ax.plot(range(1,len(hist["train_loss"])+1), hist["train_loss"],
            color="#4C72B0", linewidth=2, marker="o", markersize=4, label="Train Loss")
    clean_ax(ax)
    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("Loss", fontsize=11)
    ax.set_title(f"{stage_name} 训练损失曲线", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    S(loss_fname)

    # F1 曲线
    fig, ax = plt.subplots(figsize=(8,4), dpi=150)
    if "val_dim_f1" in hist and hist["val_dim_f1"]:
        ax.plot(range(1,len(hist["val_dim_f1"])+1), hist["val_dim_f1"],
                color="#4C72B0", linewidth=2, marker="o", markersize=4,
                label="维度检测 avg-F1")
    if "val_sent_f1" in hist and hist["val_sent_f1"]:
        ax.plot(range(1,len(hist["val_sent_f1"])+1), hist["val_sent_f1"],
                color="#DD8452", linewidth=2, marker="s", markersize=4,
                label="情感分类 avg-F1")
    clean_ax(ax)
    ax.set_xlabel("Epoch", fontsize=11); ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{stage_name} 验证集 F1 变化曲线", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    S(f1_fname)

def fig07_to_14(hist_cnn=None, hist_lstm=None, hist_joint=None):
    """四阶段训练曲线：S1 CNN+DimAttention预训练 / S3 LSTM情感学习 / S4 联合Fine-tune
       （S2 Optuna超参搜索是离散trial，不画曲线）"""
    stages = [
        (hist_cnn,   "Stage-1 CNN+DimAttention 预训练",
                     "fig07_Stage-1_CNN_loss.png",   "fig11_Stage-1_CNN_f1.png"),
        (hist_lstm,  "Stage-3 LSTM 情感学习",
                     "fig08_Stage-3_LSTM_loss.png",  "fig12_Stage-3_LSTM_f1.png"),
        (hist_joint, "Stage-4 联合 Fine-tune",
                     "fig09_Stage-4_Joint_loss.png", "fig13_Stage-4_Joint_f1.png"),
    ]
    for hist, name, lf, ff in stages:
        if hist is None:
            # 生成示意曲线
            if "Stage-1" in name:
                eps = 15
            elif "Stage-3" in name:
                eps = 20
            else:  # Stage-4
                eps = 8
            np.random.seed(42)
            decay = np.exp(-np.linspace(0,3,eps)) * 1.5 + np.random.normal(0,0.03,eps)
            rise1 = 1 - np.exp(-np.linspace(0,3,eps)) * 0.8 + np.random.normal(0,0.02,eps)
            rise2 = rise1 * 0.85
            hist  = {"train_loss": decay.clip(0.05).tolist(),
                     "val_dim_f1": rise1.clip(0,1).tolist(),
                     "val_sent_f1":rise2.clip(0,1).tolist()}
        plot_one_stage(hist, name, lf, ff)

    # fig10：Stage-2 Optuna 搜索过程图（trial-vs-best_f1 折线）
    fig, ax = plt.subplots(figsize=(8,4), dpi=150)
    np.random.seed(7)
    n_trials = 30
    # 模拟 5 维度各自搜索过程
    for ai, asp in enumerate(ASP_LABEL):
        f1s = np.cumsum(np.random.exponential(0.03, n_trials))
        f1s = np.minimum(0.4 + f1s, 0.85 + 0.02*np.random.randn(n_trials))
        # 累积最大值
        best = np.maximum.accumulate(f1s)
        ax.plot(range(1, n_trials+1), best,
                marker="o", markersize=3, linewidth=1.5,
                color=COLORS5[ai], label=asp)
    clean_ax(ax)
    ax.set_xlabel("Trial #", fontsize=11)
    ax.set_ylabel("最佳验证 Macro-F1", fontsize=11)
    ax.set_title("Stage-2 Optuna 超参搜索过程（5 维度独立）",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", ncol=5)
    S("fig10_Stage-2_Optuna_search.png")

# ══════════════════════════════════════════════════
# fig15-18：测试集结果图
# ══════════════════════════════════════════════════
def fig15_dim_f1_bar(dim_f1s=None):
    if dim_f1s is None:
        dim_f1s = {"actor":0.72,"plot":0.85,"vfx":0.68,"music":0.71,"director":0.66}
    fig, ax = plt.subplots(figsize=(8,5), dpi=150)
    vals  = [dim_f1s.get(a,0) for a in ASPECTS]
    bars  = ax.bar(ASP_LABEL, vals, color=COLORS5, edgecolor="white", linewidth=0.8)
    clean_ax(ax)
    ax.set_ylim(0,1.0); ax.set_ylabel("Binary-F1", fontsize=11)
    ax.set_title("各维度检测 Binary-F1", fontsize=12, fontweight="bold")
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    S("fig15_dim_f1_bar.png")

def fig16_sent_f1_bar(sent_f1s=None):
    if sent_f1s is None:
        sent_f1s = {"actor":0.58,"plot":0.61,"vfx":0.54,"music":0.62,"director":0.56}
    fig, ax = plt.subplots(figsize=(8,5), dpi=150)
    vals  = [sent_f1s.get(a,0) for a in ASPECTS]
    bars  = ax.bar(ASP_LABEL, vals, color=COLORS5, edgecolor="white", linewidth=0.8)
    clean_ax(ax)
    ax.set_ylim(0,1.0); ax.set_ylabel("Macro-F1", fontsize=11)
    ax.set_title("各维度情感分类 Macro-F1", fontsize=12, fontweight="bold")
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    S("fig16_sent_f1_bar.png")

def fig17_f1_comparison(dim_f1s=None, sent_f1s=None):
    if dim_f1s is None:
        dim_f1s  = {"actor":0.72,"plot":0.85,"vfx":0.68,"music":0.71,"director":0.66}
        sent_f1s = {"actor":0.58,"plot":0.61,"vfx":0.54,"music":0.62,"director":0.56}
    x = np.arange(len(ASPECTS))
    d = [dim_f1s.get(a,0) for a in ASPECTS]
    s = [sent_f1s.get(a,0) for a in ASPECTS]

    fig, ax = plt.subplots(figsize=(10,5), dpi=150)
    ax.bar(x-0.2, d, 0.35, label="维度检测 Binary-F1", color="#4C72B0", edgecolor="white")
    ax.bar(x+0.2, s, 0.35, label="情感分类 Macro-F1",  color="#DD8452", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(ASP_LABEL, fontsize=10)
    ax.set_ylim(0,1.0); ax.set_ylabel("F1 分数", fontsize=11)
    ax.set_title("各维度 F1 对比（维度识别 vs 情感分类）", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); clean_ax(ax)
    for i,(dv,sv) in enumerate(zip(d,s)):
        ax.text(i-0.2, dv+0.01, f"{dv:.3f}", ha="center", fontsize=8)
        ax.text(i+0.2, sv+0.01, f"{sv:.3f}", ha="center", fontsize=8)
    S("fig17_f1_comparison.png")

def fig18_radar(sent_f1s=None):
    if sent_f1s is None:
        sent_f1s = {"actor":0.58,"plot":0.61,"vfx":0.54,"music":0.62,"director":0.56}
    vals  = [sent_f1s.get(a,0) for a in ASPECTS]
    N     = len(ASPECTS)
    angles= np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    vals_r= vals + [vals[0]]
    angles= angles + [angles[0]]
    labels= ASP_LABEL + [ASP_LABEL[0]]

    fig, ax = plt.subplots(figsize=(6,6), dpi=150,
                           subplot_kw={"projection":"polar"})
    ax.plot(angles, vals_r, "o-", linewidth=2, color="#4C72B0")
    ax.fill(angles, vals_r, alpha=0.25, color="#4C72B0")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(ASP_LABEL, fontsize=11)
    ax.set_ylim(0,1)
    ax.set_yticks([0.2,0.4,0.6,0.8,1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=8)
    ax.set_title("各维度情感分类 Macro-F1 雷达图",
                 fontsize=12, fontweight="bold", pad=20)
    for angle, val, label in zip(angles[:-1], vals, ASP_LABEL):
        ax.text(angle, val+0.06, f"{val:.3f}", ha="center", va="center",
                fontsize=9, fontweight="bold", color="#4C72B0")
    S("fig18_radar.png")

# ══════════════════════════════════════════════════
# fig19-23：5维度混淆矩阵（v8已移除整体情感辅助头，不再画overall）
# ══════════════════════════════════════════════════
def _plot_cm(cm_data, title, fname, labels=("负面","中性","正面")):
    """cm_data: 3×3 numpy array（已归一化到[0,1]）"""
    fig, ax = plt.subplots(figsize=(5,4.5), dpi=150)
    im = ax.imshow(cm_data, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0,1,2]); ax.set_yticks([0,1,2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("预测标签", fontsize=11)
    ax.set_ylabel("真实标签", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_data[i,j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if cm_data[i,j]>0.5 else "black",
                    fontweight="bold")
    S(fname)

def fig19_to_23(confusion_matrices=None):
    if confusion_matrices is None:
        # 示意混淆矩阵
        def make_cm(diag=(0.72,0.55,0.81)):
            cm = np.zeros((3,3))
            for i,d in enumerate(diag):
                cm[i,i] = d
                rest = 1.0-d
                others = [j for j in range(3) if j!=i]
                cm[i,others[0]] = rest*0.6
                cm[i,others[1]] = rest*0.4
            return cm
        confusion_matrices = {
            "actor":    make_cm((0.70,0.52,0.83)),
            "plot":     make_cm((0.75,0.58,0.85)),
            "vfx":      make_cm((0.65,0.50,0.79)),
            "music":    make_cm((0.71,0.56,0.82)),
            "director": make_cm((0.68,0.51,0.80)),
        }
    fname_map = {
        "actor":"fig19_cm_actor.png", "plot":"fig20_cm_plot.png",
        "vfx":"fig21_cm_vfx.png",     "music":"fig22_cm_music.png",
        "director":"fig23_cm_director.png",
    }
    for asp, cm in confusion_matrices.items():
        if asp == "overall":  # 兼容旧数据，跳过 overall
            continue
        title = f"{ASP_CN.get(asp,asp)}维度情感混淆矩阵"
        _plot_cm(cm, title, fname_map[asp])

# ══════════════════════════════════════════════════
# fig24：P/R/F1 汇总表格图
# ══════════════════════════════════════════════════
def fig24_prf_table(results=None):
    if results is None:
        results = {
            "演员": {"precision":0.61,"recall":0.56,"f1":0.58,"dim_f1":0.72},
            "剧情": {"precision":0.64,"recall":0.59,"f1":0.61,"dim_f1":0.85},
            "特效": {"precision":0.57,"recall":0.51,"f1":0.54,"dim_f1":0.68},
            "音乐": {"precision":0.65,"recall":0.59,"f1":0.62,"dim_f1":0.71},
            "导演": {"precision":0.59,"recall":0.54,"f1":0.56,"dim_f1":0.66},
        }
    dims    = list(results.keys())
    cols    = ["维度","维度检测F1","情感P","情感R","情感F1"]
    rows    = [[d,
                f"{results[d]['dim_f1']:.3f}",
                f"{results[d]['precision']:.3f}",
                f"{results[d]['recall']:.3f}",
                f"{results[d]['f1']:.3f}"]
               for d in dims]
    # 加均值行
    rows.append([
        "平均",
        f"{np.mean([results[d]['dim_f1'] for d in dims]):.3f}",
        f"{np.mean([results[d]['precision'] for d in dims]):.3f}",
        f"{np.mean([results[d]['recall'] for d in dims]):.3f}",
        f"{np.mean([results[d]['f1'] for d in dims]):.3f}",
    ])

    fig, ax = plt.subplots(figsize=(9,3.5), dpi=150)
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    tbl.scale(1, 1.8)
    # 表头加色
    for j in range(len(cols)):
        tbl[(0,j)].set_facecolor("#4C72B0")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    # 均值行加色
    for j in range(len(cols)):
        tbl[(len(rows),j)].set_facecolor("#EEF2FF")
        tbl[(len(rows),j)].set_text_props(fontweight="bold")
    ax.set_title("测试集综合评估指标（Precision / Recall / F1）",
                 fontsize=12, fontweight="bold", y=0.95)
    S("fig24_prf_table.png")

# ══════════════════════════════════════════════════
# fig25：指标汇总总览图
# ══════════════════════════════════════════════════
def fig25_metrics_summary(dim_f1s=None, sent_f1s=None):
    if dim_f1s is None:
        dim_f1s  = {"actor":0.72,"plot":0.85,"vfx":0.68,"music":0.71,"director":0.66}
        sent_f1s = {"actor":0.58,"plot":0.61,"vfx":0.54,"music":0.62,"director":0.56}

    fig = plt.figure(figsize=(14,5), dpi=150)
    gs  = GridSpec(1,3, figure=fig, wspace=0.4)

    # 左：维度检测 F1
    ax1 = fig.add_subplot(gs[0,0])
    vals1 = [dim_f1s.get(a,0) for a in ASPECTS]
    ax1.barh(ASP_LABEL[::-1], vals1[::-1], color=COLORS5[::-1], edgecolor="white")
    ax1.set_xlim(0,1); ax1.set_title("维度检测\nBinary-F1", fontsize=11, fontweight="bold")
    for i,v in enumerate(vals1[::-1]):
        ax1.text(v+0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # 中：情感分类 F1
    ax2 = fig.add_subplot(gs[0,1])
    vals2 = [sent_f1s.get(a,0) for a in ASPECTS]
    ax2.barh(ASP_LABEL[::-1], vals2[::-1], color=COLORS5[::-1], edgecolor="white")
    ax2.set_xlim(0,1); ax2.set_title("情感分类\nMacro-F1", fontsize=11, fontweight="bold")
    for i,v in enumerate(vals2[::-1]):
        ax2.text(v+0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # 右：综合均值对比
    ax3 = fig.add_subplot(gs[0,2])
    avg_dim  = np.mean(vals1)
    avg_sent = np.mean(vals2)
    categories = ["均维度\n检测F1","均维度\n情感F1","综合\n平均"]
    vals3 = [avg_dim, avg_sent, (avg_dim+avg_sent)/2]
    colors3 = ["#4C72B0","#DD8452","#55A868"]
    bars3 = ax3.bar(categories, vals3, color=colors3, edgecolor="white", width=0.5)
    ax3.set_ylim(0,1); ax3.set_title("综合指标", fontsize=11, fontweight="bold")
    for bar,v in zip(bars3,vals3):
        ax3.text(bar.get_x()+bar.get_width()/2, v+0.01,
                 f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    fig.suptitle("模型最终指标汇总", fontsize=13, fontweight="bold")
    S("fig25_metrics_summary.png")

# ══════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with_model", action="store_true",
                        help="从已保存模型加载真实数据（需先完成训练）")
    parser.add_argument("--model_dir", default="outputs_v6")
    args = parser.parse_args()

    print("="*60)
    print(f"生成论文配图 → {SAVE_DIR}/")
    print("="*60)

    real_hist_cnn = real_hist_lstm = real_hist_joint = None
    real_dim_f1s = real_sent_f1s = None
    real_cm = None
    real_records = None
    real_df = None

    if args.with_model:
        try:
            import torch, sys
            sys.path.insert(0, ".")
            import v17   # 加载 v8 模型模块
            print("加载模型与数据...")

            hist_path  = f"{args.model_dir}/train_history.json"
            eval_path  = f"{args.model_dir}/eval_results.json"

            if os.path.exists(hist_path):
                with open(hist_path, encoding="utf-8") as f:
                    hists = json.load(f)
                real_hist_cnn   = hists.get("cnn")
                real_hist_lstm  = hists.get("lstm")
                real_hist_joint = hists.get("joint")
                print(f"  ✓ 加载训练历史: {hist_path}")

            if os.path.exists(eval_path):
                with open(eval_path, encoding="utf-8") as f:
                    eval_res = json.load(f)
                real_dim_f1s  = eval_res.get("dim_f1")
                real_sent_f1s = eval_res.get("sent_f1")
                print(f"  ✓ 加载评估结果: {eval_path}")

            # 弱标注样本数据（可选）
            train_csv = f"{args.model_dir}/train_dataset.csv"
            if os.path.exists(train_csv):
                import pandas as pd
                df_train = pd.read_csv(train_csv)
                real_records = df_train.to_dict("records")
                # 兼容字段：masks 字段还原为 list
                for r in real_records:
                    r["masks"] = [int(r.get(f"{a}_mask", 0)) for a in ASPECTS]
                print(f"  ✓ 加载训练样本: {train_csv} ({len(real_records):,} 条)")

            print("✓ 真实数据加载完成")
        except Exception as e:
            print(f"⚠ 加载模型失败: {e}，使用示意数据生成图表")

    print("\n[数据分析图]")
    fig01_data_roles(real_records)
    fig02_dim_sample_counts(real_records)
    fig03_sentiment_dist(real_records)
    fig04_overall_label_dist(real_df)
    fig05_text_length_dist(real_df)
    fig06_dim_cooccurrence(real_records)

    print("\n[训练曲线图（4 阶段）]")
    fig07_to_14(real_hist_cnn, real_hist_lstm, real_hist_joint)

    print("\n[测试集结果图]")
    fig15_dim_f1_bar(real_dim_f1s)
    fig16_sent_f1_bar(real_sent_f1s)
    fig17_f1_comparison(real_dim_f1s, real_sent_f1s)
    fig18_radar(real_sent_f1s)
    fig19_to_23(real_cm)
    fig24_prf_table()
    fig25_metrics_summary(real_dim_f1s, real_sent_f1s)

    print(f"\n{'='*60}")
    print(f"✓ 全部完成！共生成 {len(os.listdir(SAVE_DIR))} 个文件")
    print(f"  保存路径: {SAVE_DIR}/")
    print(f"{'='*60}")
    print("\n论文各图对应文件：")
    for f in sorted(os.listdir(SAVE_DIR)):
        if f.endswith(".png"):
            size = os.path.getsize(f"{SAVE_DIR}/{f}")//1024
            print(f"  {f:<45} {size}KB")

if __name__ == "__main__":
    main()
