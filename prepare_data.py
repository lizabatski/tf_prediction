import numpy as np
import pandas as pd
from pyfaidx import Fasta
import pyBigWig
import random
import os
from multiprocessing import Pool, cpu_count

# ============================================================
# CONFIGURATION
# ============================================================

FACTORBOOK = "data/factorbook/factorbookMotifPos.txt"
REGULATORY_BED = "data/regulatory/GM12878_regulatory.bed"
CHR1_FASTA = "data/reference/chr1.fa"
PWM_FILE = "data/factorbook/factorbookMotifPwm.txt"
CHR_NAME = "chr1"


STRUCT_DIR = "/home/ekourb/links/projects/def-majewski/ekourb/tf"
BW_PATHS = [
    f"{STRUCT_DIR}/hg19.Buckle.chr1.bw",
    f"{STRUCT_DIR}/hg19.MGW.2nd.chr1.bw",
    f"{STRUCT_DIR}/hg19.Opening.chr1.bw",
    f"{STRUCT_DIR}/hg19.ProT.2nd.chr1.bw",
    f"{STRUCT_DIR}/hg19.Roll.2nd.chr1.bw",
]

OUTPUT_DIR = "datasets_chr1"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_TFS = ["CTCF", "GATA1"]  # adjust based on what PWMs you have
PWM_SCORE_THRESHOLD_PERCENTILE = 80
WINDOW_SIZE = 30
HALF = WINDOW_SIZE // 2

random.seed(42)
np.random.seed(42)

# ============================================================
# GLOBALS (FILLED AFTER LOADING)
# ============================================================

genome_seq = None
chrom_len = None
reg = None
fb = None
pwms = None
STRUCT_ARRAYS = []  # list of np arrays, one per BW

# ============================================================
# HELPERS
# ============================================================

def one_hot_encode(seq):
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        if b in mapping:
            arr[mapping[b], i] = 1.0
    return arr

def load_pwms():
    pwms_local = {}
    with open(PWM_FILE) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 6:
                continue
            tf, L = parts[0], int(parts[1])
            a = [float(x) for x in parts[2].rstrip(",").split(",") if x]
            c = [float(x) for x in parts[3].rstrip(",").split(",") if x]
            g = [float(x) for x in parts[4].rstrip(",").split(",") if x]
            t = [float(x) for x in parts[5].rstrip(",").split(",") if x]
            pwm = np.array([a, c, g, t], dtype=np.float32)
            if pwm.shape[1] == L:
                pwms_local[tf] = pwm
    return pwms_local

def vectorized_pwm_scan(seq, log_pwm, threshold):
    """
    Your original-ish scan: loop over positions, but use numpy for scoring.
    Safer and still fast enough, especially with BigWig optimizations + MP.
    """
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    L = log_pwm.shape[1]
    seq_upper = seq.upper()
    if len(seq_upper) < L:
        return []

    idxs = np.array([mapping.get(b, -1) for b in seq_upper], dtype=np.int32)
    bad = np.where(idxs == -1)[0]

    matches = []
    for i in range(len(seq_upper) - L + 1):
        if np.any((bad >= i) & (bad < i + L)):
            continue
        window_idx = idxs[i:i+L]
        score = log_pwm[window_idx, np.arange(L)].sum()
        if score >= threshold:
            matches.append(i)
    return matches

def overlaps_with_existing(ws, we, existing_regions, min_gap=10):
    for existing_ws, existing_we in existing_regions:
        if not (we + min_gap <= existing_ws or ws >= existing_we + min_gap):
            return True
    return False

def load_bigwig_numpy(path, chrom, chrom_len):
    bw = pyBigWig.open(path)
    arr = np.array(bw.values(chrom, 0, chrom_len), dtype=np.float32)
    bw.close()
    return arr

def extract_struct_features(start, end):
    feats = []
    for arr in STRUCT_ARRAYS:
        vals = arr[start:end]
        finite = np.isfinite(vals)
        if not np.any(finite):
            feats.extend([np.nan, np.nan])
        else:
            vals_f = vals[finite]
            feats.append(float(vals_f.mean()))
            feats.append(float(vals_f.std()))
    return np.array(feats, dtype=np.float32)

def find_hard_negatives(tf, log_pwm, threshold, chip_intervals, num_needed):
    hard_negs = []
    L = log_pwm.shape[1]

    for _, row in reg.iterrows():
        if len(hard_negs) >= num_needed * 2:
            break

        s, e = int(row.start), int(row.end)
        matches = vectorized_pwm_scan(genome_seq[s:e], log_pwm, threshold)

        for m in matches:
            motif_start = s + m
            motif_end = s + m + L
            center = (motif_start + motif_end) // 2
            ws, we = center - HALF, center + HALF

            if ws < 0 or we > chrom_len:
                continue

            bound = np.any((chip_intervals[:, 0] < motif_end) &
                           (chip_intervals[:, 1] > motif_start))

            if not bound:
                hard_negs.append((ws, we, 0))
                if len(hard_negs) >= num_needed * 2:
                    break

    return hard_negs

def get_random_negatives_from_regulatory(chip_intervals, num_needed, excluded_regions):
    random_negs = []
    max_attempts = num_needed * 50
    attempts = 0

    reg_regions = []
    for _, row in reg.iterrows():
        s, e = int(row.start), int(row.end)
        if e - s >= WINDOW_SIZE:
            reg_regions.append((s, e))

    if len(reg_regions) == 0:
        print("Warning: No suitable regulatory regions found")
        return []

    while len(random_negs) < num_needed and attempts < max_attempts:
        attempts += 1

        reg_start, reg_end = random.choice(reg_regions)

        if reg_end - reg_start <= WINDOW_SIZE:
            center = (reg_start + reg_end) // 2
        else:
            center = random.randint(reg_start + HALF, reg_end - HALF)

        ws, we = center - HALF, center + HALF

        if ws < 0 or we > chrom_len:
            continue

        if ws < reg_start or we > reg_end:
            continue

        overlaps_chip = np.any((chip_intervals[:, 0] < we) &
                               (chip_intervals[:, 1] > ws))
        if overlaps_chip:
            continue

        if overlaps_with_existing(ws, we, excluded_regions):
            continue

        random_negs.append((ws, we, 0))

    if len(random_negs) < num_needed:
        print(f"Warning: only found {len(random_negs)}/{num_needed} random negatives after {attempts} attempts")

    return random_negs

# ============================================================
# MAIN TF PROCESS FUNCTION
# ============================================================

def process_tf(TF):
    print(f"[{TF}] starting...")

    if TF not in pwms:
        print(f"[{TF}] Missing PWM → skipping")
        return

    pwm = pwms[TF]
    log_pwm = np.log(pwm + 1e-10)
    L = pwm.shape[1]

    tf_chip = fb[fb["tf"] == TF]
    chip_intervals = tf_chip[["start", "end"]].to_numpy(int)
    if len(chip_intervals) == 0:
        print(f"[{TF}] No ChIP peaks → skipping")
        return

    # Threshold from random sequences
    random_scores = []
    bases = ["A", "C", "G", "T"]
    for _ in range(2000):
        seq = "".join(random.choice(bases) for _ in range(L))
        score = sum(log_pwm["ACGT".index(b), i] for i, b in enumerate(seq))
        random_scores.append(score)
    threshold = np.percentile(random_scores, PWM_SCORE_THRESHOLD_PERCENTILE)
    print(f"[{TF}] PWM threshold = {threshold:.2f}")

    # Scan regulatory regions
    print(f"[{TF}] scanning regulatory regions...")
    hits = []
    for _, row in reg.iterrows():
        s, e = int(row.start), int(row.end)
        matches = vectorized_pwm_scan(genome_seq[s:e], log_pwm, threshold)
        for m in matches:
            hits.append((s + m, s + m + L))
    print(f"[{TF}] found {len(hits)} PWM hits in regulatory regions")

    # Positives
    pos = []
    for s, e in hits:
        center = (s + e) // 2
        ws, we = center - HALF, center + HALF
        if ws < 0 or we > chrom_len:
            continue
        bound = np.any((chip_intervals[:, 0] < e) &
                       (chip_intervals[:, 1] > s))
        if bound:
            pos.append((ws, we, 1))

    n_pos = len(pos)
    if n_pos == 0:
        print(f"[{TF}] no positive samples → skipping")
        return

    print(f"[{TF}] {n_pos} positives")

    # Negatives
    n_neg_total = n_pos
    n_hard = n_neg_total // 2
    n_rand = n_neg_total - n_hard
    print(f"[{TF}] target negatives: {n_hard} hard, {n_rand} random")

    print(f"[{TF}] finding hard negatives...")
    hard_negs = find_hard_negatives(TF, log_pwm, threshold, chip_intervals, n_hard)
    if len(hard_negs) < n_hard:
        print(f"[{TF}] Warning: only found {len(hard_negs)} hard negatives")
        n_rand = n_neg_total - len(hard_negs)
    else:
        hard_negs = random.sample(hard_negs, n_hard)

    excluded_regions = [(ws, we) for ws, we, _ in pos + hard_negs]

    print(f"[{TF}] finding random negatives...")
    random_negs = get_random_negatives_from_regulatory(chip_intervals, n_rand, excluded_regions)
    if len(random_negs) < n_rand:
        print(f"[{TF}] Warning: only found {len(random_negs)} random negatives")

    all_negs = hard_negs + random_negs
    samples = pos + all_negs
    random.shuffle(samples)

    print(f"[{TF}] Final sample counts: pos={len(pos)}, hard={len(hard_negs)}, rand={len(random_negs)}")

    # Extract features
    Xs, Xstruct, Ys = [], [], []
    for ws, we, y in samples:
        seq = genome_seq[ws:we]
        feats = extract_struct_features(ws, we)
        if np.any(np.isnan(feats)):
            continue
        Xs.append(one_hot_encode(seq))
        Xstruct.append(feats)
        Ys.append(y)

    if len(Ys) == 0:
        print(f"[{TF}] After filtering, 0 samples → nothing to save")
        return

    X_seq = np.stack(Xs)
    X_struct = np.stack(Xstruct)
    y = np.array(Ys)

    out = os.path.join(OUTPUT_DIR, f"{TF.lower()}_chr1_dataset_optionC_hard_negs.npz")
    np.savez(out, X_seq=X_seq, X_struct=X_struct, y=y)
    print(f"[{TF}] saved: {out} ({len(y)} samples)")

# ============================================================
# LOAD SHARED DATA AT IMPORT TIME
# ============================================================

print("Loading genome...")
genome_seq = str(Fasta(CHR1_FASTA)[CHR_NAME]).upper()
chrom_len = len(genome_seq)
print(f"Chr1 length = {chrom_len:,}")

print("Loading regulatory regions...")
reg = pd.read_csv(REGULATORY_BED, sep="\t", header=None,
                  usecols=[0, 1, 2], names=["chrom", "start", "end"])
reg = reg[reg["chrom"] == CHR_NAME].reset_index(drop=True)
print(f"Loaded {len(reg)} regulatory regions on {CHR_NAME}")

print("Loading factorbook motif positions...")
fb = pd.read_csv(FACTORBOOK, sep="\t", header=None,
                 names=["id", "chrom", "start", "end", "tf", "score", "strand"])
fb = fb[fb["chrom"] == CHR_NAME].reset_index(drop=True)
print(f"Loaded {len(fb)} factorbook entries on {CHR_NAME}")

print("Loading PWMs...")
pwms = load_pwms()
print(f"Loaded PWMs for TFs: {list(pwms.keys())}")

print("Pre-loading BigWig tracks into memory...")
STRUCT_ARRAYS = [load_bigwig_numpy(p, CHR_NAME, chrom_len) for p in BW_PATHS]
print(f"Loaded {len(STRUCT_ARRAYS)} structural tracks")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    n_workers = min(len(TARGET_TFS), cpu_count())
    print(f"Using {n_workers} workers over TFs: {TARGET_TFS}")
    with Pool(n_workers) as pool:
        pool.map(process_tf, TARGET_TFS)
    print("Done.")
