# ParameterGolf 1.18 BPB Plan (BLACKBOXAI)
Target: val_bpb <=1.18 (beats 1.2 goal, top-20 leaderboard) on 2xA100 (~12min total).

## Status: 3/12 ✅

### 1. ✅ Record dir: `records/track_10min_16mb/2024-10-28_BLACKBOX_v1.18target/`

### 2. ✅ requirements.txt (+zstandard)

### 3. ✅ Hyperparams flags added to train_gpt.py (NUM_LAYERS=11 etc.)

### 4. ✅ LeakyReLU² MLP (MLP.forward: negative_slope=0.5)

### 5. [ ] BigramHashEmbedding (~40LOC)

### 6. [ ] XSA in CausalSelfAttention (~15LOC)

### 7. [ ] EMA+SWA (~25LOC)

### 8. [ ] GPTQ-lite int6 (~100LOC)

### 9. [ ] LZMA9 + prune (~40LOC)

### 10. [ ] Sliding eval (~30LOC)

### 11. [ ] 1GPU test

### 12. [ ] 2xA100 run
   - `mkdir -p records/track_10min_16mb/2024-xx-xx_BLACKBOX_1.18ish`
   - Copy baseline train_gpt.py there as train_gpt_v1.py

### 2. [ ] Update requirements.txt
   - Add `zstandard` for LZMA9 compat

### 3. ✅ Hyperparams flags (train_gpt.py):
   NUM_LAYERS=11 MLP_MULT=3.0 (float!) BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128 XSA_LAST_N=4 WARMDOWN_ITERS=3500 TRAIN_SEQ_LEN=2048 EVAL_SEQ_LEN=2048 EVAL_STRIDE=64

### 4. [ ] LeakyReLU² MLP (MLP.forward)

### 4. [✅] Implement LeakyReLU² MLP (~20LOC)
   - `F.leaky_relu(..., negative_slope=0.5).square()` vs `relu.square()`

### 5. [✅] Add BigramHashEmbedding class + flag (~40LOC)
   - Low-rank (1536x128) bigram pos bias after tok_emb

### 6. [ ] Add XSA (eXtra Skip Attention) in CausalSelfAttention (~15LOC)
   - Efficient GQA self-value subtract, last 4 layers

### 7. [ ] EMA(0.997)+SWA(every50 late) post-loop (~25LOC)

### 8. [ ] GPTQ-lite int6 AR self-gen calib + mixed quant (~100LOC)
   - Gen 64x2048 seqs temp=0.8 (no ext data)
   - Per-row int6 attn/mlp weights, int8 others
   - Dequant roundtrip before final eval

### 9. [ ] LZMA preset=9 + selective ±1 prune to TARGET_MB=15.9 (~40LOC)

### 10. [ ] Sliding window eval stride=64 (~30LOC)

### 11. [ ] 1GPU local test (SEED=42)
   ```
   pip install zstandard
   cd parameter-golf
   torchrun --standalone --nproc_per_node=1 train_gpt.py \
     NUM_LAYERS=11 MLP_MULT=3.0 BIGRAM_VOCAB_SIZE=1536 BIGRAM_DIM=128 \
     XSA_LAST_N=4 WARMDOWN_ITERS=3500 TRAIN_SEQ_LEN=2048 EVAL_STRIDE=64 \
     SEED=42 MAX_WALLCLOCK_SECONDS=900 VAL_LOSS_EVERY=1000
   ```
   Expect ~1.18-1.20 BPB, <16MB, <15min 1GPU.

### 12. [ ] 2xA100 supercomputer run + records
   ```
   # On supercomputer (adjust launcher)
   torchrun --nnodes=1 --nproc_per_node=2 --node_rank=0 \
     train_gpt.py WORLD_SIZE=2 SEED=42 ... (same flags)
   ```
   - Average 3 seeds (42,1337,2024)
   - Create submission.json/README.md from logs
   - PR if top-20!

**Total LOC add ~300** (baseline ~1400 -> ~1700, under 1500 strict limit via clean impl).

**Next**: Create record dir + requirements edit. Confirm before code changes?

