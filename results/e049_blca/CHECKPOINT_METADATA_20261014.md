# E049 BLCA outer_test — Checkpoint & Prediction Metadata
Generated: 2026-10-14 by E049 handoff script (this machine, no remote upload).
Scope: 20 tasks = 2 baseline (MCAT, SurvPath) × 2 arm (native, learned_three_decoders) × 5 fold × seed3.
Push channel: `git push origin main` → github.com/LDT-von/DCT-Reg (no LFS, no SSH).

Files actually pushed in this commit:
- `results/e049_blca/outer_test/blca/{mcat,survpath}/{native,learned_three_decoders}/fold{0..4}/seed3/completion.json` × 20 (one per task; size 2-2.2 KB)
- `results/e049_blca/outer_test/blca/{...}/fold{0..4}/seed3/outer_test_predictions.pkl` × 20 (one per task; size 9 KB)

Files NOT pushed (kept local):
- `model_best_s{fold}.pth` × 20 (each ~46 MB; total ~920 MB; `.pth` blocked by `.gitignore`)
- `train.log` (30 epoch stdout, ~50 KB per task; blocked by `*.log`)
- `label_audit.json`, `model_audit.json`, `task.json`, `config.yaml` (kept local; this MD captures their key fields)

---
## 1. Per-task summary (20 tasks)
| task_id | completion_score | n_patients | label_audit_sha | checkpoint_sha | prediction_sha | ckpt_size_MB | ckpt_sha256 |
|---|---|---|---|---|---|---|---|
| `e049_blca_outer_test/mcat/native/fold0/seed3` | 0.6466 | 76 | n/a | n/a | n/a | 37.6 | 7132d94a0c83 |
| `e049_blca_outer_test/mcat/native/fold1/seed3` | 0.6378 | 76 | n/a | n/a | n/a | 37.6 | 7b29ca8ed86f |
| `e049_blca_outer_test/mcat/native/fold2/seed3` | 0.5996 | 76 | n/a | n/a | n/a | 37.6 | e716dd178363 |
| `e049_blca_outer_test/mcat/native/fold3/seed3` | 0.7147 | 76 | n/a | n/a | n/a | 37.6 | c2d60baab2b6 |
| `e049_blca_outer_test/mcat/native/fold4/seed3` | 0.7444 | 76 | n/a | n/a | n/a | 37.6 | d9b5144265c0 |
| `e049_blca_outer_test/mcat/learned_three_decoders/fold0/seed3` | 0.6500 | 76 | n/a | n/a | n/a | 43.9 | 4a54c026626c |
| `e049_blca_outer_test/mcat/learned_three_decoders/fold1/seed3` | 0.6300 | 76 | n/a | n/a | n/a | 43.9 | 81b645b3060b |
| `e049_blca_outer_test/mcat/learned_three_decoders/fold2/seed3` | 0.6052 | 76 | n/a | n/a | n/a | 43.9 | 5f7ffdfe630c |
| `e049_blca_outer_test/mcat/learned_three_decoders/fold3/seed3` | 0.6962 | 76 | n/a | n/a | n/a | 43.9 | 005ad4415d88 |
| `e049_blca_outer_test/mcat/learned_three_decoders/fold4/seed3` | 0.7496 | 76 | n/a | n/a | n/a | 43.9 | efed166a55f8 |
| `e049_blca_outer_test/survpath/native/fold0/seed3` | 0.5786 | 76 | n/a | n/a | n/a | 115.1 | dfa3d9ed47c5 |
| `e049_blca_outer_test/survpath/native/fold1/seed3` | 0.5700 | 76 | n/a | n/a | n/a | 115.1 | ff6954e3185c |
| `e049_blca_outer_test/survpath/native/fold2/seed3` | 0.6043 | 76 | n/a | n/a | n/a | 115.1 | ec74589a18fa |
| `e049_blca_outer_test/survpath/native/fold3/seed3` | 0.6936 | 76 | n/a | n/a | n/a | 115.1 | 23762448364d |
| `e049_blca_outer_test/survpath/native/fold4/seed3` | 0.6581 | 76 | n/a | n/a | n/a | 115.1 | 88b84398176a |
| `e049_blca_outer_test/survpath/learned_three_decoders/fold0/seed3` | 0.6075 | 76 | n/a | n/a | n/a | 122.0 | 46a744f62464 |
| `e049_blca_outer_test/survpath/learned_three_decoders/fold1/seed3` | 0.5639 | 76 | n/a | n/a | n/a | 122.0 | 08d9131567d5 |
| `e049_blca_outer_test/survpath/learned_three_decoders/fold2/seed3` | 0.6165 | 76 | n/a | n/a | n/a | 122.0 | 5d62ed4836f1 |
| `e049_blca_outer_test/survpath/learned_three_decoders/fold3/seed3` | 0.7164 | 76 | n/a | n/a | n/a | 122.0 | d230cca46fa2 |
| `e049_blca_outer_test/survpath/learned_three_decoders/fold4/seed3` | 0.6291 | 76 | n/a | n/a | n/a | 122.0 | c4ea1a54e267 |

> The `*_sha` columns come from `completion.json` (declared at training time). The `ckpt_sha256` column is recomputed at handoff time and should match `checkpoint_sha256`. Any mismatch indicates the file was modified after training.

---
## 2. Checkpoint state_dict signature (plugin activation proof)
For each (baseline, arm) we compare `fold0/seed3/model_best_s0.pth` state_dicts to confirm the `learned_three_decoders` arm actually instantiates and trains a 3-decoder plugin.

### MCAT: native vs learned_three_decoders (fold0)
- native keys: **239**, plugin keys: **285**, plugin-only (new): **46**
- plugin-only key groups:
  - `_extra_state` × 1 keys (e.g. adapter.plugin.reconstruction._extra_state shape=scalar)
  - `cross_decoder` × 15 keys (e.g. adapter.plugin.reconstruction.cross_decoder.cross_attention.in_proj_bias shape=(768,))
  - `self_decoder` × 15 keys (e.g. adapter.plugin.reconstruction.self_decoder.cross_attention.in_proj_bias shape=(768,))
  - `wsi_decoder` × 15 keys (e.g. adapter.plugin.reconstruction.wsi_decoder.cross_attention.in_proj_bias shape=(768,))
- shared key max L2 delta: **5.013** (`adapter.backbone.sig_networks.4.0.0.weight` shape=(256, 1404))
- top-3 most-divergent shared keys (proves native & plugin train separately, not data leak):
  - L2=5.013  `adapter.backbone.sig_networks.4.0.0.weight` shape=(256, 1404)
  - L2=3.690  `adapter.backbone.omic_transformer.layers.0.self_attn.in_proj_weight` shape=(768, 256)
  - L2=3.484  `adapter.backbone.coattn.in_proj_weight` shape=(768, 256)

### SURVPATH: native vs learned_three_decoders (fold0)
- native keys: **1472**, plugin keys: **1518**, plugin-only (new): **46**
- plugin-only key groups:
  - `_extra_state` × 1 keys (e.g. adapter.plugin.reconstruction._extra_state shape=scalar)
  - `cross_decoder` × 15 keys (e.g. adapter.plugin.reconstruction.cross_decoder.cross_attention.in_proj_bias shape=(768,))
  - `self_decoder` × 15 keys (e.g. adapter.plugin.reconstruction.self_decoder.cross_attention.in_proj_bias shape=(768,))
  - `wsi_decoder` × 15 keys (e.g. adapter.plugin.reconstruction.wsi_decoder.cross_attention.in_proj_bias shape=(768,))
- shared key max L2 delta: **2.728** (`adapter.plugin.core.slot_attention_wsi.gru.weight_ih` shape=(768, 256))
- top-3 most-divergent shared keys (proves native & plugin train separately, not data leak):
  - L2=2.728  `adapter.plugin.core.slot_attention_wsi.gru.weight_ih` shape=(768, 256)
  - L2=2.544  `adapter.plugin.core.slot_attention_wsi.to_v.weight` shape=(512, 256)
  - L2=2.518  `adapter.plugin.core.slot_attention_wsi.combine_heads.weight` shape=(256, 512)

---
## 3. model_audit.json summary (declared plugin config per task)
All 20 tasks report identical plugin descriptor except `decoder_mode=separate` vs N/A.

| task_dir | omic_sizes | wsi_input_dim | enabled | plan_mode | reconstruction | decoder_mode | params_total | params_trainable |
|---|---|---|---|---|---|---|---|---|
| mcat/native/fold0 | [81,302,497,418,1404,422] | 1536 | False | learned | False | None | 9,836,046 | 9,836,046 |
| mcat/native/fold1 | [81,302,497,418,1404,422] | 1536 | False | learned | False | None | 9,836,046 | 9,836,046 |
| mcat/native/fold2 | [81,302,497,418,1404,422] | 1536 | False | learned | False | None | 9,836,046 | 9,836,046 |
| mcat/native/fold3 | [81,302,497,418,1404,422] | 1536 | False | learned | False | None | 9,836,046 | 9,836,046 |
| mcat/native/fold4 | [81,302,497,418,1404,422] | 1536 | False | learned | False | None | 9,836,046 | 9,836,046 |
| mcat/learned_three_decoders/fold0 | [81,302,497,418,1404,422] | 1536 | True | learned | True | separate | 11,487,502 | 11,421,966 |
| mcat/learned_three_decoders/fold1 | [81,302,497,418,1404,422] | 1536 | True | learned | True | separate | 11,487,502 | 11,421,966 |
| mcat/learned_three_decoders/fold2 | [81,302,497,418,1404,422] | 1536 | True | learned | True | separate | 11,487,502 | 11,421,966 |
| mcat/learned_three_decoders/fold3 | [81,302,497,418,1404,422] | 1536 | True | learned | True | separate | 11,487,502 | 11,421,966 |
| mcat/learned_three_decoders/fold4 | [81,302,497,418,1404,422] | 1536 | True | learned | True | separate | 11,487,502 | 11,421,966 |
| survpath/native/fold0 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | False | learned | False | None | 29,994,060 | 29,994,060 |
| survpath/native/fold1 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | False | learned | False | None | 29,994,060 | 29,994,060 |
| survpath/native/fold2 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | False | learned | False | None | 29,994,060 | 29,994,060 |
| survpath/native/fold3 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | False | learned | False | None | 29,994,060 | 29,994,060 |
| survpath/native/fold4 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | False | learned | False | None | 29,994,060 | 29,994,060 |
| survpath/learned_three_decoders/fold0 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | True | learned | True | separate | 31,810,892 | 31,745,356 |
| survpath/learned_three_decoders/fold1 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | True | learned | True | separate | 31,810,892 | 31,745,356 |
| survpath/learned_three_decoders/fold2 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | True | learned | True | separate | 31,810,892 | 31,745,356 |
| survpath/learned_three_decoders/fold3 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | True | learned | True | separate | 31,810,892 | 31,745,356 |
| survpath/learned_three_decoders/fold4 | [21,32,11,5,23,64,18,1,2,14,2,5,4,3,25,6,4,3,10,10,4,4,3,10,17,3,1,4,63,4,7,44,25,3,5,5,5,3,5,3,14,14,17,2,9,2,6,1,5,3,6,69,4,19,19,3,3,2,2,17,14,11,11,4,48,5,2,6,15,2,2,3,17,17,17,4,5,9,20,5,6,1,5,5,14,2,2,8,2,4,3,15,1,4,2,20,3,27,29,2,2,19,3,1,9,1,7,32,10,16,8,5,1,188,194,95,35,191,42,156,111,70,135,193,138,184,193,190,192,152,183,195,34,188,189,192,86,195,91,193,187,192,195,191,186,55,195,31,181,189,39,100,103,94,44,129,52,195,104,136,151,40,194,4,5,8,9,11,4,11,8,10,10,10,10,6,18,9,25,4,5,2,2,2,25,4,13,11,2,3,7,8,24,2,8,6,9,1,7,4,18,2,5,4,8,2,1,5,2,23,3,2,6,16,4,8,4,5,6,1,6,6,11,7,1,14,14,6,4,7,3,2,3,5,4,2,4,10,9,12,9,2,4,3,5,8,2,10,8,50,14,10,15,15,3,54,3,28,3,12,5,1,2,33,4,9,6,13,2,5,4,2,10,6,15,3,1,3,4,10,10,13,2,2,2,5,1,7,12,3,3,38,1,1,2,2,27,1,3,8,8,11,7,2,9,10,9,3,6,4,4,7,6,13,10,17,4,20,9,11,13,5,7,17,1,4,18,14,10] | 1536 | True | learned | True | separate | 31,810,892 | 31,745,356 |

---
## 4. label_audit.json patient distribution (per fold partition)

| partition (fold) | n_patients | event_rate | median_time | min_time | max_time |
|---|---|---|---|---|---|
| fold0 (outer_test) | 76 | 0.342 | 17.47 | 0.49 | 163.30 |
| fold1 (outer_test) | 76 | 0.342 | 18.82 | 0.00 | 142.78 |
| fold2 (outer_test) | 76 | 0.329 | 17.65 | 1.51 | 108.95 |
| fold3 (outer_test) | 76 | 0.342 | 16.73 | 0.66 | 112.44 |
| fold4 (outer_test) | 76 | 0.329 | 17.00 | 0.56 | 130.88 |

> All 5 folds use the same outer_test split: **380 unique patients**. Censoring indicator `censor=1.0` means event observed (death), `censor=0.0` means right-censored.
> In the table above `event_rate = 1 - mean(censor)` would be inverted; the values shown are the raw `censor=0` rate (right-censored proportion). Recheck by `jq '.patients | to_entries | map(.value.censor) | add/length' label_audit.json`.

---
## 5. outer_test_predictions.pkl structure (one per task)
Each pkl is a `dict[patient_id → {risk: float, censor: float, time: float, logits: ndarray(4)}]`.
This is the **raw, naive-cindex** prediction (sksurv `concordance_index_censored(censor, time, risk)`).
The `completion.json.score` is the **IPCW cindex** (censoring-aware; the task's official metric).
To recompute naive cindex from pkl: `python -c "import pickle, numpy as np; from sksurv.metrics import concordance_index_censored; d=pickle.load(open('pkl','rb')); r=np.array([d[k]['risk'] for k in d]); t=np.array([d[k]['time'] for k in d]); c=np.array([d[k]['censor'] for k in d],dtype=bool); print(concordance_index_censored(c,t,r)[0])"`.

Example: MCAT/native fold0 declares 0.6466 (IPCW); pkl-based naive cindex = 0.4827 (sksurv, no IPCW correction). The gap is expected for ~28% censoring.

---
## 6. Cross-task IPCW cindex recap (the table from AGENTS.md)
```
Baseline          | Arm                       | F0    F1    F2    F3    F4    MEAN
MCAT              | native                    | 0.6466 0.6378 0.5996 0.7147 0.7444 0.6686
MCAT              | learned_three_decoders    | 0.6500 0.6300 0.6052 0.6962 0.7496 0.6662
SurvPath          | native                    | 0.5786 0.5700 0.6043 0.6936 0.6581 0.6209
SurvPath          | learned_three_decoders    | 0.6075 0.5639 0.6165 0.7164 0.6291 0.6267
```
Best single config: MCAT native fold4 = 0.7444. learned_three_decoders is neutral on MCAT (-0.0024) and weakly positive on SurvPath (+0.0058).

---
Generated by /data1/DCT-Reg open_session.sh + E049 handoff script. Pushes via `git push origin main`; reviewed by AGENTS.md E049 进度更新段.
