"""
================================================================
 豆瓣电影评论多维度情感分析 v6.FINAL (字符级编码版)
 —— CNN维度检测 + 5路独立LSTM（维度条件注意力）+ Optuna贝叶斯超参
================================================================

【核心改进】
  ① 彻底移除整体情感头（ov_head），只保留5路独立LSTM
  ② SingleDimLSTMHead 升级为维度条件注意力：
       attention 打分时注入维度 embedding，
       让模型自己学会聚焦相关词位（深度学习版"分句剔除"）
  ③ 改进弱标注：结合电影中位数识别反讽/脑残粉样本送人工
  ④ Stage-2 LSTM 只用有维度标注样本训练（不被 los 污染）
  ⑤ 修复内存泄漏（gc.collect + empty_cache + del lstm/opt）
  ⑥ 彻底落实字符级编码（Character-level）提升模型泛化鲁棒性

使用：
  python v6_final.py --stage all
  python v6_final.py --stage train --bayes_trials 30
================================================================
"""

import os, re, json, warnings, argparse, pathlib, gc
from collections import Counter, defaultdict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
import jieba
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
jieba.setLogLevel("WARN")
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    print("[警告] optuna 未安装，贝叶斯优化将跳过。pip install optuna")

# ════════════════════════════════════════════════════════════
# 0. 全局常量
# ════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_LEN = 80
VOCAB_SIZE = 20002
EMBED_DIM = 64
EPOCHS_CNN = 15
EPOCHS_LSTM = 20
BAYES_TRIALS = 30
SEED = 42
OUT_DIR = "outputs_v6"
os.makedirs(OUT_DIR, exist_ok=True)

# 电影均分"偏低"阈值
LOW_MOVIE_MEDIAN = 3.0

torch.manual_seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print(f"运行设备: {DEVICE}  |  Optuna: {OPTUNA_OK}")
print("=" * 60)

# ════════════════════════════════════════════════════════════
# 1. 维度关键词词典
# ════════════════════════════════════════════════════════════
ASPECT_KEYWORDS = {
    "actor": ["演员", "演技", "表演", "主演", "配角", "男主", "女主",
              "男演员", "女演员", "表现力", "代入感", "飙戏", "对手戏", "演绎", "颜值",
              "明星", "实力派", "老戏骨", "影帝", "影后", "选角", "卡司", "阵容", "戏骨",
              "扮演", "出演"],
    "plot": ["剧情", "故事", "情节", "剧本", "编剧", "叙事", "节奏", "逻辑", "结局", "悬念",
             "反转", "高潮", "铺垫", "伏笔", "主线", "支线", "漏洞", "套路", "脑洞",
             "设定", "架构", "台词", "对白", "改编", "翻拍", "烧脑", "悬疑", "开头", "开场",
             "片头", "片尾", "桥段", "故事线", "背景设定", "主题", "内容",
             "文戏", "武戏", "动作戏",
             "角色", "人物", "角色塑造", "人物塑造", "人物刻画", "人物设定",
             "角色设定", "人物弧线", "人物成长", "人设", "配角戏", "群像"],
    "vfx": ["特效", "视效", "画面", "视觉", "场面",
            "特技", "CG", "CGI", "3D", "IMAX",
            "场景", "布景", "美术", "置景",
            "色调", "调色", "光影", "构图",
            "服化道", "服装", "道具", "造型", "航拍", "VFX"],
    "music": ["音乐", "配乐", "原声", "OST", "插曲", "主题曲", "背景音乐", "BGM", "音效",
              "声音", "混音", "作曲", "旋律", "歌曲", "音响", "配音", "声效",
              "交响", "片尾曲", "原声带"],
    "director": ["导演", "执导", "风格", "手法", "表达", "深度", "内涵", "格局", "水准",
                 "诚意", "敷衍", "圈钱", "蒙太奇", "剪辑", "后期", "制作",
                 "摄影", "镜头", "镜头语言", "运镜", "调度",
                 "长镜头", "短镜头", "镜头切换",
                 "近景", "远景", "全景", "特写", "中景", "大全景",
                 "景深", "焦距", "手持", "转场", "叙事节奏"],
}

ASPECTS = list(ASPECT_KEYWORDS.keys())
NUM_ASP = len(ASPECTS)
ASP_IDX = {a: i for i, a in enumerate(ASPECTS)}
ASP_CN = {"actor": "演员", "plot": "剧情", "vfx": "特效", "music": "音乐", "director": "导演"}
ASP_CN_LIST = [ASP_CN[a] for a in ASPECTS]

NUM_SENT = 3  # 0=负面  1=中性  2=正面
SENT_CN = {0: "负面", 1: "中性", 2: "正面"}


# ════════════════════════════════════════════════════════════
# 2. 星级 → 情感标签
# ════════════════════════════════════════════════════════════
def star2label(star0):
    """star0 已减1（0~4）"""
    if star0 <= 1: return 0
    if star0 == 2: return 1
    return 2


# ════════════════════════════════════════════════════════════
# 3. 停用词 & 分词工具
# ════════════════════════════════════════════════════════════
STOPWORDS = set([
    "的", "地", "得", "之", "乎", "者", "也", "我", "你", "他", "她", "它",
    "我们", "你们", "他们", "她们", "它们", "自己", "这", "那", "这个", "那个",
    "这些", "那些", "这里", "那里", "此", "该", "什么", "怎么", "怎样", "为什么",
    "哪里", "哪儿", "谁", "多少", "哪", "于", "由", "从", "向", "往", "把", "被", "对",
    "为", "因", "所", "若", "虽", "如", "即", "则", "且", "以", "以及",
    "啊", "吧", "呢", "哦", "嗯", "哎", "呀", "嘛", "唉", "而已", "已经", "曾经",
    "将", "刚", "正在", "接着", "然后", "随后", "同时",
])


# ----------------- 字符级工具 (用于模型训练与推理) -----------------
def tokenize_all_char(texts):
    result = []
    for text in tqdm(texts, desc="字符级切分", ncols=80):
        toks = [c for c in str(text) if c.strip()]
        result.append(toks)
    return result


def build_vocab_char(texts, vocab_size=VOCAB_SIZE):
    counter = Counter()
    for text in tqdm(texts, desc="建字符词表", ncols=80):
        counter.update(c for c in str(text) if c.strip())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for c, _ in counter.most_common(vocab_size - 2):
        vocab[c] = len(vocab)
    return vocab


def encode(toks, vocab, max_len=MAX_LEN):
    ids = [vocab.get(c, 1) for c in toks[:max_len]]
    return ids + [0] * (max_len - len(ids))


# ----------------- 词级工具 (仅用于弱标注匹配) -----------------
def tokenize_all(texts):
    result = []
    for text in tqdm(texts, desc="分词匹配", ncols=80):
        toks = [w for w in jieba.cut(str(text)) if w.strip() and w not in STOPWORDS]
        result.append(toks)
    return result


# ════════════════════════════════════════════════════════════
# 4. 新弱标注策略
# ════════════════════════════════════════════════════════════
def compute_movie_stats(df, movie_col="Movie_Name_CN", star_col="Star"):
    stats = {}
    for movie, grp in df.groupby(movie_col, sort=False):
        stars = grp[star_col].values.astype(float)
        stats[movie] = {
            "median": float(np.median(stars)),
            "mean": float(stars.mean()),
            "count": len(stars),
        }
    return stats


def count_dimensions(tokens):
    hit_dims, hit_kws = set(), defaultdict(list)
    tok_set = set(tokens)
    for asp, kws in ASPECT_KEYWORDS.items():
        for kw in kws:
            if kw in tok_set:
                hit_dims.add(asp)
                hit_kws[asp].append(kw)
    return hit_dims, dict(hit_kws)


def weak_annotate(df, movie_stats, tokenized_cache,
                  movie_col="Movie_Name_CN", star_col="star0"):
    records_train = []
    records_manual = []
    records_irony = []
    records_los = []

    for i in tqdm(range(len(df)), desc="弱标注", ncols=80):
        row = df.iloc[i]
        text = str(row["Comment"])
        star0 = int(row[star_col])
        star_real = star0 + 1
        tokens = tokenized_cache[i]  # 词级别的token，仅用于维度匹配
        movie = row.get(movie_col, "")

        hit_dims, hit_kws = count_dimensions(tokens)
        dim_count = len(hit_dims)
        hit_list = list(hit_dims)

        movie_med = movie_stats.get(movie, {}).get("median", 3.0)
        low_movie = (movie_med <= LOW_MOVIE_MEDIAN)

        is_low = (star_real <= 2)
        is_mid = (star_real == 3)
        is_high = (star_real >= 4)

        labels = [-1] * NUM_ASP
        masks = [0] * NUM_ASP

        base = {
            "idx": i, "text": text, "movie": movie,
            "star": star_real, "movie_median": movie_med,
            "low_movie": low_movie,
            "dim_count": dim_count,
            "hit_dims": hit_list, "hit_kws": hit_kws,
        }

        # ── 0个维度 ──
        if dim_count == 0:
            if is_low:
                records_los.append({**base, "role": "los_SIN"})
                records_train.append({**base, "labels": labels, "masks": masks, "role": "los_SIN"})
            elif is_high and not low_movie:
                records_los.append({**base, "role": "los_LOVE"})
                records_train.append({**base, "labels": labels, "masks": masks, "role": "los_LOVE"})
            else:
                hint = ("电影口碑差却打高分，0维→疑似反讽" if low_movie and is_high
                        else "3星0维→表意不明")
                base["irony_hint"] = hint
                records_irony.append(base)
                records_manual.append(base)
                records_train.append({**base, "labels": labels, "masks": masks, "role": "manual_pending"})
            continue

        # ── 1个维度 ──
        if dim_count == 1:
            dim = hit_list[0]
            ai = ASP_IDX[dim]
            if is_low:
                labels[ai] = 0;
                masks[ai] = 1
                records_train.append({**base, "labels": labels, "masks": masks, "role": "auto_1dim"})
            elif is_high:
                labels[ai] = 2;
                masks[ai] = 1
                records_train.append({**base, "labels": labels, "masks": masks, "role": "auto_1dim"})
            else:
                hint = ("电影口碑差1维中分→反讽" if low_movie else "3星1维→方向不明")
                base["irony_hint"] = hint
                records_irony.append(base)
                records_manual.append(base)
                records_train.append({**base, "labels": labels, "masks": masks, "role": "manual_pending"})
            continue

        # ── ≥2个维度 ──
        if is_low:
            for dim in hit_list:
                ai = ASP_IDX[dim];
                labels[ai] = 0;
                masks[ai] = 1
            records_train.append({**base, "labels": labels, "masks": masks, "role": "auto_multi_neg"})
        elif is_high:
            for dim in hit_list:
                ai = ASP_IDX[dim];
                labels[ai] = 2;
                masks[ai] = 1
            records_train.append({**base, "labels": labels, "masks": masks, "role": "auto_multi_pos"})
        else:
            # 中分：前半负面，后半正面
            half = max(1, dim_count // 2)
            for k, dim in enumerate(hit_list):
                ai = ASP_IDX[dim]
                labels[ai] = 0 if k < half else 2
                masks[ai] = 1
            records_train.append({**base, "labels": labels, "masks": masks, "role": "auto_multi_mid"})

    roles = Counter(r.get("role", "") for r in records_train)
    total = len(records_train)
    print(f"\n[弱标注完成] 共 {total:,} 条")
    print(f"  反讽/待人工: {len(records_manual):,}  los: {len(records_los):,}")
    print("  训练集组成:")
    for role, cnt in sorted(roles.items(), key=lambda x: -x[1]):
        print(f"    {role:<25}: {cnt:>8,}  ({cnt / total * 100:.1f}%)")

    return records_train, records_manual, records_irony, records_los


def merge_manual_labels(records_train, manual_labeled_path):
    if not os.path.exists(manual_labeled_path):
        return records_train
    idx2pos = {r["idx"]: j for j, r in enumerate(records_train)}
    merged = 0
    with open(manual_labeled_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("skipped", False): continue
            idx = obj["idx"]
            if idx not in idx2pos: continue
            rec = records_train[idx2pos[idx]]
            if rec["role"] != "manual_pending": continue
            for dim_str, lbl in obj["labels"].items():
                if dim_str in ASP_IDX:
                    ai = ASP_IDX[dim_str]
                    rec["labels"][ai] = int(lbl)
                    rec["masks"][ai] = 1
            rec["role"] = "manual_labeled"
            merged += 1
    print(f"[merge_manual] 合并人工标注: {merged} 条")
    return records_train


def save_datasets(records_train, records_manual, records_irony, records_los):
    def _to_df(records):
        rows = []
        for r in records:
            row = {"text": r["text"], "movie": r.get("movie", ""),
                   "star": r.get("star", ""), "movie_median": r.get("movie_median", ""),
                   "low_movie": r.get("low_movie", ""), "role": r.get("role", "")}
            for ai, asp in enumerate(ASPECTS):
                row[f"{asp}_mask"] = r.get("masks", [0] * NUM_ASP)[ai]
                row[f"{asp}_sent"] = r.get("labels", [-1] * NUM_ASP)[ai]
            rows.append(row)
        return pd.DataFrame(rows)

    print("\n[保存数据集]")
    for records, name in [(records_train, "train_dataset"),
                          (records_manual, "manual_queue"),
                          (records_irony, "irony_samples"),
                          (records_los, "los_labels")]:
        if not records: continue
        df_out = _to_df(records)
        df_out.to_csv(f"{OUT_DIR}/{name}.csv", index=False, encoding="utf-8-sig")
        print(f"  ✓ {name}.csv  ({len(df_out):,} 条)")


def load_train_dataset(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    records = []
    for _, row in df.iterrows():
        masks = [int(row[f"{asp}_mask"]) for asp in ASPECTS]
        labels = [int(row[f"{asp}_sent"]) for asp in ASPECTS]
        records.append({"text": str(row["text"]), "masks": masks,
                        "labels": labels, "role": str(row.get("role", ""))})
    return records


# ════════════════════════════════════════════════════════════
# 5. PyTorch Dataset
# ════════════════════════════════════════════════════════════
class MultiAspectDataset(Dataset):
    def __init__(self, records, X_encoded):
        self.records = records
        self.X_encoded = X_encoded

    def __len__(self): return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        x = torch.tensor(self.X_encoded[i], dtype=torch.long)
        dim_presence = torch.tensor([float(r["masks"][j]) for j in range(NUM_ASP)])
        asp_lbl = torch.tensor([max(0, r["labels"][j]) for j in range(NUM_ASP)], dtype=torch.long)
        asp_mask = torch.tensor(r["masks"], dtype=torch.float)
        return x, dim_presence, asp_lbl, asp_mask


# ════════════════════════════════════════════════════════════
# 6. CNN 编码器
# ════════════════════════════════════════════════════════════
class CNNEncoder(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 filters=32, kernels=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, filters, k, padding=k // 2),
                nn.BatchNorm1d(filters), nn.ReLU(), nn.Dropout(dropout * 0.5)
            ) for k in kernels
        ])
        self.cnn_out_dim = filters * len(kernels)
        self.ln = nn.LayerNorm(self.cnn_out_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        e = self.emb(x)
        xt = e.permute(0, 2, 1)
        co = [c(xt) for c in self.convs]
        ml = min(o.size(2) for o in co)
        seq_out = self.ln(torch.cat([o[:, :, :ml] for o in co], 1).permute(0, 2, 1))
        pool_out = self.drop(seq_out.max(1).values)
        return pool_out, seq_out


# ════════════════════════════════════════════════════════════
# 7. CNN 维度检测头 & DimAttention
# ════════════════════════════════════════════════════════════
class CNNDimHead(nn.Module):
    def __init__(self, in_dim, num_asp=NUM_ASP):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2), nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(in_dim // 2, num_asp)
        )

    def forward(self, pool_out):
        return self.fc(pool_out)


class DimAttention(nn.Module):
    def __init__(self, cnn_out_dim, attn_dim=None):
        super().__init__()
        attn_dim = attn_dim or cnn_out_dim
        self.key_proj = nn.Linear(cnn_out_dim, attn_dim, bias=False)
        self.queries = nn.Parameter(torch.randn(NUM_ASP, attn_dim))
        nn.init.xavier_uniform_(self.queries.unsqueeze(0))
        self.scale = attn_dim ** -0.5
        self.attn_dim = attn_dim

    def forward(self, seq_out, pad_mask=None):
        K = self.key_proj(seq_out)  # [B, seq, attn_dim]
        scores = torch.einsum("ad,bsd->bas", self.queries, K) * self.scale
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        attn_w = torch.softmax(scores, dim=-1)  # [B, NUM_ASP, seq]
        attn_out = torch.einsum("bas,bsd->bad", attn_w, seq_out)  # [B, NUM_ASP, cnn_out]
        return attn_out, attn_w


# ════════════════════════════════════════════════════════════
# 8. 单维度 LSTM 情感头
# ════════════════════════════════════════════════════════════
class SingleDimLSTMHead(nn.Module):
    def __init__(self, cnn_out_dim, num_layers=1, hidden_size=128, dropout=0.3, bidirectional=True):
        super().__init__()
        lstm_in = cnn_out_dim * 2  # seq_out + dim_vec 拼接
        self.lstm = nn.LSTM(
            lstm_in, hidden_size, num_layers=num_layers,
            batch_first=True, bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0
        )
        ld = hidden_size * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(ld)
        self.attn = nn.Sequential(nn.Linear(ld, ld // 2), nn.Tanh(), nn.Linear(ld // 2, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(ld, ld // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ld // 2, NUM_SENT)
        )

    def forward(self, dim_vec, seq_out):
        B, seq_len, D = seq_out.shape
        dv_exp = dim_vec.unsqueeze(1).expand(-1, seq_len, -1)  # [B, seq, D]
        lstm_in = torch.cat([seq_out, dv_exp], dim=-1)  # [B, seq, 2D]
        h, _ = self.lstm(lstm_in)
        h = self.ln(h)
        a = torch.softmax(self.attn(h), dim=1)
        ctx = self.drop((h * a).sum(1))
        return self.fc(ctx)


# ════════════════════════════════════════════════════════════
# 9. 完整模型
# ════════════════════════════════════════════════════════════
class HybridModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM,
                 cnn_filters=64, cnn_kernels=(3, 4, 5), cnn_dropout=0.3,
                 attn_dim=None, lstm_configs=None, dim_threshold=0.5):
        super().__init__()
        self.dim_threshold = dim_threshold
        self.cnn_encoder = CNNEncoder(vocab_size, embed_dim, cnn_filters, cnn_kernels, cnn_dropout)
        cnn_out = self.cnn_encoder.cnn_out_dim
        self.dim_head = CNNDimHead(cnn_out)
        self.dim_attention = DimAttention(cnn_out, attn_dim)

        if lstm_configs is None:
            lstm_configs = [{"num_layers": 1, "hidden_size": 64, "dropout": 0.3, "bidirectional": True}] * NUM_ASP
        self.lstm_heads = nn.ModuleList([
            SingleDimLSTMHead(cnn_out, **cfg) for cfg in lstm_configs
        ])

    def forward(self, x, dim_labels_gt=None, use_gt_labels=True):
        pad_mask = (x == 0)
        pool_out, seq_out = self.cnn_encoder(x)
        dim_logits = self.dim_head(pool_out)

        attn_out, attn_w = self.dim_attention(seq_out, pad_mask[:, :seq_out.size(1)])

        if use_gt_labels and dim_labels_gt is not None:
            lbl4lstm = dim_labels_gt
        else:
            lbl4lstm = (torch.sigmoid(dim_logits) >= self.dim_threshold).float()

        sent_list = []
        for ai in range(NUM_ASP):
            dv = attn_out[:, ai, :]
            out_i = self.lstm_heads[ai](dv, seq_out)
            sent_list.append(out_i)
        sent_logits = torch.stack(sent_list, dim=1)

        return dim_logits, sent_logits, attn_w

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            dim_logits, sent_logits, attn_w = self.forward(x, use_gt_labels=False)
            dim_probs = torch.sigmoid(dim_logits)
            sent_preds = sent_logits.argmax(-1)
        results = []
        for b in range(x.size(0)):
            rec = {}
            for ai, asp in enumerate(ASPECTS):
                prob = float(dim_probs[b, ai])
                if prob >= self.dim_threshold:
                    sid = int(sent_preds[b, ai])
                    rec[asp] = {"present": True, "sentiment": SENT_CN[sid],
                                "sentiment_id": sid, "prob": round(prob, 3),
                                "attn": attn_w[b, ai].cpu().numpy().tolist()}
                else:
                    rec[asp] = {"present": False, "prob": round(prob, 3)}
            results.append(rec)
        return results, attn_w.cpu().numpy()


# ════════════════════════════════════════════════════════════
# 10. 损失函数
# ════════════════════════════════════════════════════════════
class HybridLoss(nn.Module):
    def __init__(self, alpha_dim=0.3, alpha_sent=0.7, label_smooth=0.1):
        super().__init__()
        self.alpha_dim = alpha_dim
        self.alpha_sent = alpha_sent
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce_sent = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smooth,
                                           weight=torch.tensor([2.0, 1.5, 1.0], device=DEVICE))

    def forward(self, dim_logits, sent_logits, dim_mask, asp_lbl, asp_mask):
        L_dim = self.bce(dim_logits, dim_mask.float()).mean()
        L_sent = torch.tensor(0.0, device=dim_logits.device)
        n_valid = 0
        for ai in range(NUM_ASP):
            l = self.ce_sent(sent_logits[:, ai, :], asp_lbl[:, ai])
            m = asp_mask[:, ai]
            n = m.sum()
            if n > 0:
                L_sent += (l * m).sum() / n
                n_valid += 1
        if n_valid > 0:
            L_sent /= n_valid
        return self.alpha_dim * L_dim + self.alpha_sent * L_sent, {"L_dim": L_dim.item(), "L_sent": L_sent.item()}


# ════════════════════════════════════════════════════════════
# 11. 评估
# ════════════════════════════════════════════════════════════
def evaluate(model, loader, dim_threshold=0.5):
    model.eval()
    all_dp, all_dt = [], []
    asp_preds = [[] for _ in range(NUM_ASP)]
    asp_trues = [[] for _ in range(NUM_ASP)]

    with torch.no_grad():
        for x, dim_mask, asp_lbl, asp_mask in loader:
            x = x.to(DEVICE)
            dim_logits, sent_logits, _ = model(x, use_gt_labels=False)
            dp = (torch.sigmoid(dim_logits).cpu() >= dim_threshold).float()
            all_dp.append(dp.numpy());
            all_dt.append(dim_mask.numpy())
            sp = sent_logits.argmax(-1).cpu()
            for ai in range(NUM_ASP):
                valid = (asp_mask[:, ai] == 1).nonzero(as_tuple=True)[0]
                if len(valid):
                    asp_preds[ai].extend(sp[valid, ai].tolist())
                    asp_trues[ai].extend(asp_lbl[valid, ai].tolist())

    all_dp = np.vstack(all_dp);
    all_dt = np.vstack(all_dt)
    dim_f1s = {asp: f1_score(all_dt[:, ai], all_dp[:, ai], zero_division=0) for ai, asp in enumerate(ASPECTS)}
    sent_f1s = {
        asp: (f1_score(asp_trues[ai], asp_preds[ai], average="macro", zero_division=0) if asp_preds[ai] else 0.0)
        for ai, asp in enumerate(ASPECTS)}
    return {"dim_f1": dim_f1s, "sent_f1": sent_f1s,
            "avg_dim_f1": np.mean(list(dim_f1s.values())),
            "avg_sent_f1": np.mean(list(sent_f1s.values()))}


# ════════════════════════════════════════════════════════════
# 12. 训练
# ════════════════════════════════════════════════════════════
def train_stage(model, train_loader, val_loader, epochs, lr,
                stage="joint", model_name="model",
                alpha_dim=0.3, alpha_sent=0.7,
                params_override=None, param_groups=None):
    loss_fn = HybridLoss(alpha_dim, alpha_sent).to(DEVICE)
    if param_groups is not None:
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    else:
        params = params_override or list(model.parameters())
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_f1, best_state, no_improve = 0.0, None, 0
    history = {"train_loss": [], "val_dim_f1": [], "val_sent_f1": []}

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_name}|{stage}] ep{ep:03d}", ncols=100, leave=False)

        for i_b, (x, dim_mask, asp_lbl, asp_mask) in enumerate(pbar, 1):
            x, dim_mask = x.to(DEVICE), dim_mask.to(DEVICE)
            asp_lbl, asp_mask = asp_lbl.to(DEVICE), asp_mask.to(DEVICE)
            optimizer.zero_grad()

            dim_logits, sent_logits, _ = model(x, use_gt_labels=True)

            if stage == "cnn":
                loss = loss_fn.alpha_dim * loss_fn.bce(dim_logits, dim_mask.float()).mean()
            elif stage == "lstm":
                loss, _ = loss_fn(dim_logits.detach(), sent_logits, dim_mask, asp_lbl, asp_mask)
            else:
                loss, _ = loss_fn(dim_logits, sent_logits, dim_mask, asp_lbl, asp_mask)

            loss.backward()
            if param_groups is None:
                nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if i_b % 50 == 0:
                torch.cuda.empty_cache()

        scheduler.step()
        metrics = evaluate(model, val_loader)
        combined = 0.5 * metrics["avg_dim_f1"] + 0.5 * metrics["avg_sent_f1"]
        history["train_loss"].append(epoch_loss / len(train_loader))
        history["val_dim_f1"].append(metrics["avg_dim_f1"])
        history["val_sent_f1"].append(metrics["avg_sent_f1"])

        if ep == 1 and stage == "joint":
            print(f"    [Stage-4 ep1 检查] dim_F1={metrics['avg_dim_f1']:.4f} sent_F1={metrics['avg_sent_f1']:.4f}")
            if metrics["avg_dim_f1"] < 0.05:
                print("    ⚠ dim_F1 骤降（可能CNN被破坏），降低学习率继续...")
                for pg in optimizer.param_groups:
                    pg["lr"] *= 0.1

        if combined > best_f1:
            best_f1, best_state, no_improve = combined, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1

        if stage == "joint" and no_improve >= 3:
            print(f"  [{model_name}] 早停（连续3轮无提升，ep={ep}）")
            break

        if ep % 5 == 0 or ep == 1:
            print(f"  [{model_name}|{stage}] ep{ep:03d} loss={epoch_loss / len(train_loader):.4f} "
                  f"dim_F1={metrics['avg_dim_f1']:.4f} sent_F1={metrics['avg_sent_f1']:.4f}")

    if best_state: model.load_state_dict(best_state)
    print(f"  [{model_name}] 最优 combined_F1={best_f1:.4f}")
    return model, best_f1, history


# ════════════════════════════════════════════════════════════
# 13. Optuna 贝叶斯超参优化
# ════════════════════════════════════════════════════════════
def bayes_optimize_lstm(asp_idx, asp_name, records_train, X_encoded,
                        cnn_encoder, dim_attention, n_trials=BAYES_TRIALS):
    if not OPTUNA_OK:
        return {"num_layers": 1, "hidden_size": 128, "dropout": 0.3, "bidirectional": True}

    print(f"\n  [Bayes] [{ASP_CN[asp_name]}] LSTM超参搜索 ({n_trials} trials)...")
    masks_array = np.array([r["masks"][asp_idx] for r in records_train], dtype=np.uint8)
    valid_idx = np.where(masks_array == 1)[0].tolist()

    if len(valid_idx) < 50:
        print(f"    ⚠ 有效样本过少({len(valid_idx)})，使用默认超参")
        return {"num_layers": 1, "hidden_size": 128, "dropout": 0.3, "bidirectional": True}

    tr_idx, va_idx = train_test_split(valid_idx, test_size=0.2, random_state=SEED)
    lo_tr = DataLoader(MultiAspectDataset([records_train[i] for i in tr_idx], [X_encoded[i] for i in tr_idx]),
                       batch_size=128, shuffle=True, num_workers=0)
    lo_va = DataLoader(MultiAspectDataset([records_train[i] for i in va_idx], [X_encoded[i] for i in va_idx]),
                       batch_size=256, shuffle=False, num_workers=0)

    cnn_out = cnn_encoder.cnn_out_dim

    def objective(trial):
        cfg = {
            "num_layers": trial.suggest_int("num_layers", 1, 2),
            "hidden_size": trial.suggest_int("hidden_size", 32, 128, step=32),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "bidirectional": trial.suggest_categorical("bidirectional", [True, False]),
        }
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        lstm = SingleDimLSTMHead(cnn_out, **cfg).to(DEVICE)
        opt = torch.optim.AdamW(lstm.parameters(), lr=lr, weight_decay=1e-4)
        ce = nn.CrossEntropyLoss()

        cnn_encoder.eval();
        dim_attention.eval()
        best_f1 = 0.0
        for _ in range(8):
            lstm.train()
            for x, dim_mask, asp_lbl, asp_mask in lo_tr:
                x = x.to(DEVICE)
                lbl_i = asp_lbl[:, asp_idx].to(DEVICE)
                msk_i = asp_mask[:, asp_idx].to(DEVICE)
                if msk_i.sum() == 0: continue
                with torch.no_grad():
                    _, seq_out = cnn_encoder(x)
                    pad_mask = (x == 0)[:, :seq_out.size(1)]
                    attn_out, _ = dim_attention(seq_out, pad_mask)
                dv = attn_out[:, asp_idx, :]
                logit = lstm(dv, seq_out)
                loss = (ce(logit, lbl_i) * msk_i).sum() / (msk_i.sum() + 1e-9)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(lstm.parameters(), 5.0)
                opt.step()

            lstm.eval();
            preds, trues = [], []
            with torch.no_grad():
                for x, dim_mask, asp_lbl, asp_mask in lo_va:
                    x = x.to(DEVICE)
                    valid = (asp_mask[:, asp_idx] == 1).nonzero(as_tuple=True)[0]
                    if not len(valid): continue
                    _, seq_out = cnn_encoder(x)
                    pad_mask = (x == 0)[:, :seq_out.size(1)]
                    attn_out, _ = dim_attention(seq_out, pad_mask)
                    dv = attn_out[:, asp_idx, :]
                    pred = lstm(dv, seq_out).argmax(1).cpu()
                    preds.extend(pred[valid].tolist())
                    trues.extend(asp_lbl[valid, asp_idx].tolist())
            if preds:
                best_f1 = max(best_f1, f1_score(trues, preds, average="macro", zero_division=0))

        del lstm, opt
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return best_f1

    def safe_objective(trial):
        try:
            return objective(trial)
        except Exception as e:
            print(f"\n    [Trial {trial.number} 失败] {type(e).__name__}: {e}")
            raise

    DEFAULT_CFG = {"num_layers": 1, "hidden_size": 128, "dropout": 0.3, "bidirectional": True}
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    try:
        study.optimize(safe_objective, n_trials=1, show_progress_bar=False)
    except Exception as e:
        print(f"    ⚠ 热身trial异常: {e}")
        return DEFAULT_CFG

    if n_trials > 1:
        try:
            study.optimize(safe_objective, n_trials=n_trials - 1, show_progress_bar=False, catch=(RuntimeError,))
        except Exception as e:
            pass

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed: return DEFAULT_CFG
    best = study.best_params
    cfg = {k: best[k] for k in ("num_layers", "hidden_size", "dropout", "bidirectional")}
    print(f"    ✓ {ASP_CN[asp_name]} 最优: {cfg} F1={study.best_value:.4f}")
    return cfg


# ════════════════════════════════════════════════════════════
# 14. 完整训练流程
# ════════════════════════════════════════════════════════════
def run_training(records_train, X_encoded, vocab, n_bayes_trials=BAYES_TRIALS):
    all_idx = list(range(len(records_train)))
    tr_idx, va_idx = train_test_split(all_idx, test_size=0.15, random_state=SEED)

    def make_loader(idx_list, shuffle=True, bs=BATCH_SIZE):
        return DataLoader(
            MultiAspectDataset([records_train[i] for i in idx_list], [X_encoded[i] for i in idx_list]),
            batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    train_loader = make_loader(tr_idx)
    val_loader = make_loader(va_idx, shuffle=False)

    # ════════════════════════════════════════════════════════════
    # 断点续训：先把 vocab 存下来（每阶段都可能依赖）
    # ════════════════════════════════════════════════════════════
    with open(f"{OUT_DIR}/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    ckpt_s1 = f"{OUT_DIR}/stage1_checkpoint.pt"
    ckpt_s2 = f"{OUT_DIR}/stage2_checkpoint.pt"
    ckpt_s2_cfg = f"{OUT_DIR}/stage2_lstm_configs.json"
    ckpt_s3 = f"{OUT_DIR}/stage3_checkpoint.pt"
    hist_s1_path = f"{OUT_DIR}/stage1_history.json"
    hist_s3_path = f"{OUT_DIR}/stage3_history.json"

    # ───── Stage-1：CNN + DimAttention ─────
    if os.path.exists(ckpt_s1):
        print("\n" + "=" * 60 + "\n[跳过] Stage-1 已有检查点，直接加载\n" + "=" * 60)
        model = HybridModel(vocab_size=len(vocab)).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_s1, map_location=DEVICE))
        with open(hist_s1_path, "r", encoding="utf-8") as f:
            hist_cnn = json.load(f)
        print(f"  ✓ 加载: {ckpt_s1}")
    else:
        print("\n" + "=" * 60 + "\nStage-1: CNN 维度检测 + DimAttention 预训练\n" + "=" * 60)
        model = HybridModel(vocab_size=len(vocab)).to(DEVICE)
        cnn_params = (list(model.cnn_encoder.parameters()) + list(model.dim_head.parameters()) + list(
            model.dim_attention.parameters()))
        model, _, hist_cnn = train_stage(model, train_loader, val_loader, epochs=EPOCHS_CNN, lr=1e-3, stage="cnn",
                                         model_name="S1-CNN", alpha_dim=1.0, alpha_sent=0.0, params_override=cnn_params)
        torch.save(model.state_dict(), ckpt_s1)
        with open(hist_s1_path, "w", encoding="utf-8") as f:
            json.dump(hist_cnn, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Stage-1 检查点已保存: {ckpt_s1}")

    # ───── Stage-2：Optuna 超参搜索 ─────
    if os.path.exists(ckpt_s2) and os.path.exists(ckpt_s2_cfg):
        print("\n" + "=" * 60 + "\n[跳过] Stage-2 已有检查点，直接加载\n" + "=" * 60)
        with open(ckpt_s2_cfg, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        best_cfgs = [cfg_dict[asp] for asp in ASPECTS]
        model = HybridModel(vocab_size=len(vocab), lstm_configs=best_cfgs).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_s2, map_location=DEVICE))
        print(f"  ✓ 加载: {ckpt_s2}")
    else:
        print("\n" + "=" * 60 + "\nStage-2: Optuna 贝叶斯超参优化（5路LSTM）\n" + "=" * 60)
        best_cfgs = []
        for ai, asp in enumerate(ASPECTS):
            cfg = bayes_optimize_lstm(ai, asp, records_train, X_encoded, model.cnn_encoder, model.dim_attention,
                                      n_trials=n_bayes_trials)
            best_cfgs.append(cfg)
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        print("\n  用最优LSTM超参重建模型...")
        new_model = HybridModel(vocab_size=len(vocab), lstm_configs=best_cfgs).to(DEVICE)
        new_model.cnn_encoder.load_state_dict(model.cnn_encoder.state_dict())
        new_model.dim_head.load_state_dict(model.dim_head.state_dict())
        new_model.dim_attention.load_state_dict(model.dim_attention.state_dict())
        model = new_model

        torch.save(model.state_dict(), ckpt_s2)
        with open(ckpt_s2_cfg, "w", encoding="utf-8") as f:
            json.dump({asp: best_cfgs[ai] for ai, asp in enumerate(ASPECTS)}, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Stage-2 检查点已保存: {ckpt_s2}")

    # ───── Stage-3：LSTM 情感学习 ─────
    if os.path.exists(ckpt_s3):
        print("\n" + "=" * 60 + "\n[跳过] Stage-3 已有检查点，直接加载\n" + "=" * 60)
        model.load_state_dict(torch.load(ckpt_s3, map_location=DEVICE))
        with open(hist_s3_path, "r", encoding="utf-8") as f:
            hist_lstm = json.load(f)
        print(f"  ✓ 加载: {ckpt_s3}")
    else:
        print("\n" + "=" * 60 + "\nStage-3: LSTM 情感学习\n" + "=" * 60)
        sent_idx = [i for i in tr_idx if records_train[i].get("role", "") in (
        "auto_1dim", "auto_multi_neg", "auto_multi_pos", "auto_multi_mid", "manual_labeled")]
        if len(sent_idx) < 100: sent_idx = tr_idx
        sent_loader = make_loader(sent_idx)
        lstm_params = list(model.lstm_heads.parameters())
        model, _, hist_lstm = train_stage(model, sent_loader, val_loader, epochs=EPOCHS_LSTM, lr=5e-4, stage="lstm",
                                          model_name="S3-LSTM", alpha_dim=0.0, alpha_sent=1.0, params_override=lstm_params)
        torch.save(model.state_dict(), ckpt_s3)
        with open(hist_s3_path, "w", encoding="utf-8") as f:
            json.dump(hist_lstm, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Stage-3 检查点已保存: {ckpt_s3}")

    # ───── Stage-4：联合 Fine-tune ─────
    print("\n" + "=" * 60 + "\nStage-4: 联合 Fine-tune（差异化学习率）\n" + "=" * 60)
    joint_param_groups = [
        {"params": list(model.cnn_encoder.parameters()), "lr": 2e-5},
        {"params": list(model.dim_head.parameters()), "lr": 2e-5},
        {"params": list(model.dim_attention.parameters()), "lr": 5e-5},
        {"params": list(model.lstm_heads.parameters()), "lr": 1e-4},
    ]
    model, _, hist_joint = train_stage(model, train_loader, val_loader, epochs=EPOCHS_FINETUNE, lr=1e-4, stage="joint",
                                       model_name="S4-Joint", alpha_dim=0.1, alpha_sent=0.9,
                                       param_groups=joint_param_groups)

    # ───── 最终保存 ─────
    torch.save(model.state_dict(), f"{OUT_DIR}/model.pt")
    with open(f"{OUT_DIR}/lstm_configs.json", "w", encoding="utf-8") as f:
        json.dump({asp: best_cfgs[ai] for ai, asp in enumerate(ASPECTS)}, f, ensure_ascii=False, indent=2)

    # 保存训练历史（供 generate_figures.py --with_model 使用）
    with open(f"{OUT_DIR}/train_history.json", "w", encoding="utf-8") as f:
        json.dump({"cnn": hist_cnn, "lstm": hist_lstm, "joint": hist_joint},
                  f, ensure_ascii=False, indent=2)

    # 保存最终评估结果
    final_metrics = evaluate(model, val_loader)
    with open(f"{OUT_DIR}/eval_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "dim_f1": final_metrics["dim_f1"],
            "sent_f1": final_metrics["sent_f1"],
            "avg_dim_f1": float(final_metrics["avg_dim_f1"]),
            "avg_sent_f1": float(final_metrics["avg_sent_f1"]),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 模型/词表/LSTM配置/训练历史/评估结果已保存到 {OUT_DIR}/")
    print(f"  （阶段检查点 stage*_checkpoint.pt 保留，下次重训前可手动删除以重新训练）")
    return model, best_cfgs, hist_cnn, hist_lstm, hist_joint, val_loader


# ════════════════════════════════════════════════════════════
# 15. 推理 & 评估 (支持字符级预测及可视化)
# ════════════════════════════════════════════════════════════
def _predict_one(text, model, vocab):
    """对单条文本推理，返回结构化结果"""
    toks = [c for c in str(text) if c.strip()]
    x = torch.tensor([encode(toks, vocab)], dtype=torch.long).to(DEVICE)
    results, attn_w = model.predict(x)
    return results[0], attn_w[0], toks


def _print_result(text, res, attn_w, toks, verbose=True):
    SENT_EMOJI = {0: "👎 负面", 1: "😐 中性", 2: "👍 正面"}
    SENT_COLOR = {0: "\033[91m", 1: "\033[93m", 2: "\033[92m"}
    RESET = "\033[0m"

    print(f"\n{'─' * 60}")
    print(f"  输入：{text}")
    print(f"{'─' * 60}")

    any_found = False
    for ai, asp in enumerate(ASPECTS):
        info = res[asp]
        if info["present"]:
            any_found = True
            sid = info["sentiment_id"]
            prob = info["prob"]
            sent = SENT_EMOJI.get(sid, str(sid))
            col = SENT_COLOR.get(sid, "")

            attn = attn_w[ai]
            top_idx = attn.argsort()[::-1][:5]
            top_words = []
            for j in top_idx:
                if j < len(toks) and attn[j] > 0.02:
                    top_words.append(f"{toks[j]}({attn[j]:.2f})")
            top_str = "  →  " + "  ".join(top_words[:3]) if top_words else ""

            print(f"  {ASP_CN[asp]:4s}  {col}{sent}{RESET}  [置信度 {prob:.0%}]{top_str}")
        else:
            if verbose:
                print(f"  {ASP_CN[asp]:4s}  ── 未提及  [检测概率 {info['prob']:.0%}]")

    if not any_found:
        print("  （未检测到任何维度关键词，可能是整体评价）")
    print(f"{'─' * 60}")


def demo(model, vocab, sample_texts):
    print("\n" + "=" * 60 + "\n推理示例（含注意力可视化）\n" + "=" * 60)
    for text in sample_texts:
        res, attn_w, toks = _predict_one(text, model, vocab)
        _print_result(text, res, attn_w, toks, verbose=True)


def print_metrics(model, val_loader):
    m = evaluate(model, val_loader)
    print("\n" + "=" * 60 + "\n最终评估\n" + "=" * 60)
    print("\n【CNN 维度识别 F1】")
    for asp in ASPECTS: print(f"  {ASP_CN[asp]:4s}: {m['dim_f1'][asp]:.4f}")
    print(f"  平均: {m['avg_dim_f1']:.4f}")
    print("\n【LSTM 维度情感 F1（含DimAttention）】")
    for asp in ASPECTS: print(f"  {ASP_CN[asp]:4s}: {m['sent_f1'][asp]:.4f}")
    print(f"  平均: {m['avg_sent_f1']:.4f}")


# ════════════════════════════════════════════════════════════
# 16. 主程序
# ════════════════════════════════════════════════════════════
def _load_model_for_infer(model_dir):
    vocab_path = os.path.join(model_dir, "vocab.json")
    model_path = os.path.join(model_dir, "model.pt")
    cfg_path = os.path.join(model_dir, "lstm_configs.json")

    if not os.path.exists(vocab_path): raise FileNotFoundError(f"找不到词表: {vocab_path}")
    if not os.path.exists(model_path): raise FileNotFoundError(f"找不到模型: {model_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    lstm_configs = None
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        lstm_configs = [cfg_dict[asp] for asp in ASPECTS]

    model = HybridModel(vocab_size=len(vocab), lstm_configs=lstm_configs).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, vocab


def run_infer(model_dir=OUT_DIR, single_text=None):
    model, vocab = _load_model_for_infer(model_dir)
    if single_text is not None:
        res, attn_w, toks = _predict_one(single_text, model, vocab)
        _print_result(single_text, res, attn_w, toks)
        return
    print("\n" + "=" * 60 + "\n  豆瓣电影评论多维度情感分析  ·  交互推理\n" + "=" * 60)
    print("  特殊命令: q=退出  batch=批量模式  verbose=切换输出  help=帮助")
    verbose = True
    while True:
        try:
            text = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text: continue
        if text.lower() in ("q", "quit", "exit", "退出"): break
        if text.lower() == "help":
            print("  q/quit/exit=退出  batch=批量模式  verbose on/off=显隐未提及维度")
            continue
        if text.lower().startswith("verbose"):
            verbose = not (len(text.split()) >= 2 and text.lower().split()[1] == "off")
            print(f"  已切换为{'详细' if verbose else '简洁'}模式")
            continue
        if text.lower() == "batch":
            batch_texts = []
            while True:
                try:
                    line = input("  > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not line: break
                batch_texts.append(line)
            for bt in batch_texts:
                res, attn_w, toks = _predict_one(bt, model, vocab)
                _print_result(bt, res, attn_w, toks, verbose=verbose)
            continue
        res, attn_w, toks = _predict_one(text, model, vocab)
        _print_result(text, res, attn_w, toks, verbose=verbose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["annotate", "train", "all", "infer"])
    parser.add_argument("--manual_labeled", default="manual_queue_labeled.jsonl")
    parser.add_argument("--bayes_trials", type=int, default=BAYES_TRIALS)
    parser.add_argument("--model_dir", default=OUT_DIR)
    parser.add_argument("--text", default=None)
    args = parser.parse_args()

    if args.stage == "infer":
        run_infer(model_dir=args.model_dir, single_text=args.text)
        exit(0)

    _dir = pathlib.Path(__file__).parent.resolve()
    _csv = next((p for p in [_dir / "DMSC.csv", _dir / "dmsc.csv"] if p.exists()), None)
    if _csv is None: raise FileNotFoundError("找不到 DMSC.csv，请放在脚本同目录")

    df = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            df = pd.read_csv(str(_csv), encoding=enc, on_bad_lines="skip", low_memory=False)
            print(f"编码 {enc} 读取成功，共 {len(df):,} 行");
            break
        except Exception:
            pass
    if df is None: raise RuntimeError("所有编码均失败")

    df = df.dropna(subset=["Comment", "Star"])
    df["Comment"] = df["Comment"].astype(str)
    df["Star"] = df["Star"].astype(int).clip(1, 5)
    df["star0"] = df["Star"] - 1
    df = df[df["Comment"].str.len() >= 3].reset_index(drop=True)
    movie_col = ("Movie_Name_CN" if "Movie_Name_CN" in df.columns else
                 "movie" if "movie" in df.columns else df.columns[0])
    print(f"使用电影列: {movie_col}  共 {df[movie_col].nunique()} 部电影")

    if args.stage in ("annotate", "all"):
        print("\n" + "=" * 60 + "\n弱标注流程 (使用词级匹配)\n" + "=" * 60)
        tokenized_cache = tokenize_all(df["Comment"].tolist())  # 使用词级别匹配关键词
        movie_stats = compute_movie_stats(df, movie_col=movie_col, star_col="Star")
        medians = [s["median"] for s in movie_stats.values()]
        low_cnt = sum(1 for m in medians if m <= LOW_MOVIE_MEDIAN)
        print(f"  共 {len(medians)} 部电影  |  口碑偏低(≤{LOW_MOVIE_MEDIAN}): "
              f"{low_cnt} 部 ({low_cnt / len(medians) * 100:.1f}%)")
        records_train, records_manual, records_irony, records_los = \
            weak_annotate(df, movie_stats, tokenized_cache,
                          movie_col=movie_col, star_col="star0")
        records_train = merge_manual_labels(records_train, args.manual_labeled)
        save_datasets(records_train, records_manual, records_irony, records_los)

    if args.stage in ("train", "all"):
        print("\n" + "=" * 60 + "\n加载数据 & 构建字符词表\n" + "=" * 60)
        if args.stage == "train":
            records_train = load_train_dataset(f"{OUT_DIR}/train_dataset.csv")
            texts_for_vocab = [r["text"] for r in records_train]
        else:
            texts_for_vocab = df["Comment"].tolist()

        vocab = build_vocab_char(texts_for_vocab)

        print("编码整句序列 (字符级)...")
        all_toks = (tokenize_all_char(texts_for_vocab) if args.stage == "train"
                    else tokenize_all_char(df["Comment"].tolist()))

        X_all = [encode(toks, vocab) for toks in tqdm(all_toks, desc="整句编码", ncols=80)]

        if args.stage == "train":
            X_encoded = X_all
        else:
            X_encoded = [X_all[r["idx"]] for r in records_train]

        model, best_cfgs, h_cnn, h_lstm, h_joint, val_loader = \
            run_training(records_train, X_encoded, vocab,
                         n_bayes_trials=args.bayes_trials)

        print_metrics(model, val_loader)
        demo(model, vocab, [
            "演技太棒了，但剧情有点拖沓，特效做得很用心",
            "配乐很加分，导演手法成熟",
            "完全是圈钱之作，毫无诚意",
            "还行吧",
        ])