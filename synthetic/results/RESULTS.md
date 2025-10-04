# INITIAL RUN


C:\omega\omega>python -m cli analyze-cond ^
More?   --train corpora/stdmap/stdmap_train.txt ^
More?   --holdout corpora/stdmap/stdmap_tail.txt ^
More?   --mode words --label-mode ib ^
More?   --n 3 --k_ctx 5 --ib-clusters 32 ^
More?   --stride 1 --seed 101 ^
More?   --bootstrap 2000 --alpha 0.05 ^
More?   --ib-max-contexts 6000 ^
More?   --out-json runs/stdmap_ctx3_cl16_s101.json
 {
  "H_base": 1.841926414935292,
  "H_ib": 1.835295518295441,
  "H_hash": 1.8876574868954934,
  "delta_ib": {
    "mean": -0.006630896639851236,
    "lo95": -0.008736799613007926,
    "hi95": -0.004688104311969831,
    "p_le_0": 1.0
  },
  "delta_hash": {
    "mean": 0.04573107196020136,
    "lo95": 0.0427917233845274,
    "hi95": 0.04869156768997402,
    "p_le_0": 0.0
  },
  "k_ctx": 5,
  "ib_clusters": 32,
  "stride": 1,
  "label_mode": "ib",
  "null_label": "C-1",
  "n_clusters": 32,
  "n_gram_order": 3,
  "coverage_train": {
    "full_frac": 0.9988380675730375,
    "backoff_frac": 0.0011619324269625468,
    "null_frac": 2.381009071644563e-05
  },
  "coverage_test": {
    "full_frac": 0.9550308350463914,
    "backoff_frac": 0.04496916495360853,
    "null_frac": 5.555864214678593e-05
  },
  "notes": "Bits/token. \u0394<0 means conditional helps. Hash control preserves label histogram."
}

[Ω] summary (first variant):
  tag=true  ΔI_pred≈-0.006631  Δhash≈0.045731  SCR_lzma≈2.957e-07  energy=22424
  
  
# RND SEED CHANGE  
  
  
python -m cli analyze-cond ^
  --train corpora/stdmap/stdmap_train.txt ^
  --holdout corpora/stdmap/stdmap_tail.txt ^
  --mode words --label-mode ib ^
  --n 3 --k_ctx 5 --ib-clusters 32 ^
  --ib-max-contexts 6000 ^
  --stride 1 --seed 333 ^
  --bootstrap 2000 --alpha 0.05 ^
  --out-json runs/stdmap_ctx3_k5_cl32_ctx6k_s333.json


{
  "H_base": 1.841926414935292,
  "H_ib": 1.835295518295441,
  "H_hash": 1.8876574868954934,
  "delta_ib": {
    "mean": -0.006630896639851236,
    "lo95": -0.008736799613007926,
    "hi95": -0.004688104311969831,
    "p_le_0": 1.0
  },
  "delta_hash": {
    "mean": 0.04573107196020136,
    "lo95": 0.0427917233845274,
    "hi95": 0.04869156768997402,
    "p_le_0": 0.0
  },
  "k_ctx": 5,
  "ib_clusters": 32,
  "stride": 1,
  "label_mode": "ib",
  "null_label": "C-1",
  "n_clusters": 32,
  "n_gram_order": 3,
  "coverage_train": {
    "full_frac": 0.9988380675730375,
    "backoff_frac": 0.0011619324269625468,
    "null_frac": 2.381009071644563e-05
  },
  "coverage_test": {
    "full_frac": 0.9550308350463914,
    "backoff_frac": 0.04496916495360853,
    "null_frac": 5.555864214678593e-05
  },
  "notes": "Bits/token. \u0394<0 means conditional helps. Hash control preserves label histogram."
}

[Ω] summary (first variant):
  tag=true  ΔI_pred≈-0.006631  Δhash≈0.045731  SCR_lzma≈2.957e-07  energy=22424

C:\omega\omega>

# SHUFFLE

python -m cli analyze-cond ^
  --train corpora/stdmap/stdmap_shuf.txt ^
  --holdout corpora/stdmap/stdmap_tail_B9.txt ^
  --mode words --label-mode ib ^
  --n 3 --k_ctx 5 --ib-clusters 32 ^
  --ib-max-contexts 6000 --stride 1 --seed 101 ^
  --bootstrap 800 --alpha 0.05 ^
  --out-json runs/stdmap_control_shuf.json
  
  
{
  "H_base": 3.362268446079975,
  "H_ib": 3.362282157343671,
  "H_hash": 3.3620256778514452,
  "delta_ib": {
    "mean": 1.3711263696132968e-05,
    "lo95": -2.0953953271695157e-05,
    "hi95": 3.923412019163766e-05,
    "p_le_0": 0.20125
  },
  "delta_hash": {
    "mean": -0.00024276822852975459,
    "lo95": -0.0003557315870372208,
    "hi95": -0.0001302606821159593,
    "p_le_0": 1.0
  },
  "k_ctx": 5,
  "ib_clusters": 32,
  "stride": 1,
  "label_mode": "ib",
  "null_label": "C-1",
  "n_clusters": 32,
  "n_gram_order": 3,
  "coverage_train": {
    "full_frac": 0.12685063930093574,
    "backoff_frac": 0.8731493606990642,
    "null_frac": 2.381009071644563e-05
  },
  "coverage_test": {
    "full_frac": 8.889382743485749e-05,
    "backoff_frac": 0.9999111061725652,
    "null_frac": 5.555864214678593e-05
  },
  "notes": "Bits/token. \u0394<0 means conditional helps. Hash control preserves label histogram."
}

[Ω] summary (first variant):
  tag=true  ΔI_pred≈0.000014  Δhash≈-0.000243  SCR_lzma≈-4.483e-10  energy=30584



# BIG BOOTSTRAP

C:\omega\omega>  python -m cli analyze-cond ^
More?   --train corpora/stdmap/stdmap_train.txt ^
More?   --holdout corpora/stdmap/stdmap_tail.txt ^
More?   --mode words --label-mode ib ^
More?   --n 3 --k_ctx 5 --ib-clusters 32 ^
More?   --ib-max-contexts 6000 ^
More?   --stride 1 --seed 333 ^
More?   --bootstrap 7000 --alpha 0.05 ^
More?   --out-json runs/stdmap_ctx3_k5_cl32_ctx6k_s333.json
 {
  "H_base": 1.841926414935292,
  "H_ib": 1.835295518295441,
  "H_hash": 1.8876574868954934,
  "delta_ib": {
    "mean": -0.006630896639851236,
    "lo95": -0.008680056425824756,
    "hi95": -0.004588640743194529,
    "p_le_0": 1.0
  },
  "delta_hash": {
    "mean": 0.04573107196020136,
    "lo95": 0.04284545831556682,
    "hi95": 0.048625966298846066,
    "p_le_0": 0.0
  },
  "k_ctx": 5,
  "ib_clusters": 32,
  "stride": 1,
  "label_mode": "ib",
  "null_label": "C-1",
  "n_clusters": 32,
  "n_gram_order": 3,
  "coverage_train": {
    "full_frac": 0.9988380675730375,
    "backoff_frac": 0.0011619324269625468,
    "null_frac": 2.381009071644563e-05
  },
  "coverage_test": {
    "full_frac": 0.9550308350463914,
    "backoff_frac": 0.04496916495360853,
    "null_frac": 5.555864214678593e-05
  },
  "notes": "Bits/token. \u0394<0 means conditional helps. Hash control preserves label histogram."
}

[Ω] summary (first variant):
  tag=true  ΔI_pred≈-0.006631  Δhash≈0.045731  SCR_lzma≈2.957e-07  energy=22424