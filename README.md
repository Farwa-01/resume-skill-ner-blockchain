# Resume Skill Extraction with RoBERTa NER and Blockchain-Enabled Data Integrity

Code accompanying the paper **"Secure and Scalable Skill Extraction from Real-World Resumes using RoBERTa-Based NER with Blockchain-Enabled Data Integrity"** (Faria Karamat, Farwa Nawaz, Shawal Khaliq; submitted to IJDIIC).

The repository has two parts:

1. **`train_ner.py`**: fine-tunes `roberta-base` for skill extraction (BIO tags `O` / `B-SKILL` / `I-SKILL`) and produces every table and figure in the Results section.
2. **`blockchain/`**: the Ethereum smart contract that anchors SHA-256 hashes of skill profiles, plus the scripts that produced the gas, latency and throughput numbers in Table 5a.

## Results (validation split, 498 resumes)

| Precision | Recall | F1 | Token accuracy |
|---|---|---|---|
| 0.640 | 0.666 | 0.653 | 0.910 |

Entity-level metrics use `seqeval`. Final run values: P 0.6400, R 0.6657, F1 0.6526, accuracy 0.9096. Seed is fixed to 42.

## Data

The dataset is the public Kaggle corpus **NER Annotated CVs**: <https://www.kaggle.com/datasets/mehyarmlaweh/ner-annotated-cvs>. It is not redistributed here. Download it and point the script at the folder of annotated JSON files (`ResumesJsonAnnotated`).

5,029 files are read; 58 have no usable annotation and are dropped, leaving 4,971 resumes, split 90:10 (random state 42) into 4,473 training and 498 validation resumes.

## Running the NER pipeline

```bash
pip install -r requirements.txt
export JSON_DIR=/path/to/ResumesJsonAnnotated   # annotated resume JSON files
export WORK_DIR=outputs                          # checkpoints, models and figures go here
python train_ner.py
```

A GPU is strongly recommended (the reported run used a single Kaggle/Colab GPU with FP16). The script runs, in order: data loading and BIO tagging, three-stage training (5 + 6 + 3 epochs; learning rates 2e-5, 5e-6, 1e-6; effective batch size 32), the confusion matrix and per-class accuracy, the dataset-scale ablation (50 / 500 / 1,000 / 2,000 / 3,000 / 4,473 resumes), the learning curve, and a results summary written to `results_summary.json`. The script is also laid out as cells, so it can be pasted cell by cell into Colab or Kaggle.

**Note on warmup.** The code asks for a learning-rate warmup (`warmup_ratio`), but the `transformers` version used for the reported results did not support that argument, so it was skipped (the log prints `warmup_ratio not supported`). The reported results therefore use linear decay with no warmup. Newer versions may apply warmup and give slightly different numbers.

**Environment.** Package versions are not pinned. Results can shift slightly across library and GPU versions even with a fixed seed. Run `pip freeze` in your environment to record exact versions.

## Blockchain component

`blockchain/contracts/SkillVerification.sol` is a Solidity 0.8.24 contract with:

- `StoreProfile(candidateId, profileHash, metadataURI)`: registrar only; records cannot be overwritten.
- `VerifyProfile(candidateId, profileHash)`: compares a submitted hash with the stored one (authorized callers only).
- `FetchProfile(candidateId)`: returns the stored hash, metadata pointer and timestamp (authorized callers only).
- `GrantAccess(address)` / `RevokeAccess(address)`: registrar only.

Only the hash and a metadata pointer are stored on-chain, never the resume text or the extracted skills.

```bash
cd blockchain
npm install
node scripts/compile.js                                  # compiles with solc (optimizer, 200 runs)
HARDHAT_NETWORK=hardhat node scripts/benchmark.js        # writes build/benchmark_results.json
```

The benchmark runs on a local Hardhat Network, not a public chain. Measured values (200 profiles): `StoreProfile` about 161,000 gas, `VerifyProfile` about 29,000 gas, `FetchProfile` about 38,000 gas, deployment about 631,000 gas, end-to-end store-and-verify latency about 2.5 ms. Mainnet or testnet latency and cost would be different. Full numbers are in `blockchain/build/benchmark_results.json`.

## Limitations

Single validation split (no cross-validation); baselines in the paper are positioned from the literature, not re-trained here; annotation quality is inherited from the public dataset. See Section 5 of the paper.

## License

MIT, see `LICENSE`.
