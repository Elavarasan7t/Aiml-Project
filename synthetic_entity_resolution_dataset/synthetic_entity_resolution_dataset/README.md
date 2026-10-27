# Synthetic Business Entity-Resolution Dataset

Everything in this folder is **synthetically generated** by `generate_dataset.py` (seeded with `random.seed(42)` and
`numpy.random.seed(42)`, so every run reproduces the same files byte-for-byte). No real business database, geocoder or
external file is used. Names, streets, postal codes and landmarks are invented; city/state names are generic public
geography used only as text.

## 1. Purpose
Develop and test an ML pipeline (blocking → feature engineering → classifier → threshold tuning) that decides whether
records from different sources describe the **same real-world business**, before running it on the official hackathon data.

## 2. Layout
```
dataset/train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
dataset/test/   test_source1.tsv   test_source2.tsv   test_source3.tsv        (no labels)
evaluation/     test_ground_truth.tsv   (PRIVATE - local scoring only)
                audit_train.tsv / audit_test.tsv   (PRIVATE - hidden master_entity_id + noise tags for error analysis)
ml/             training_pairs.tsv
generate_dataset.py
```
All files are **tab-separated, UTF-8** (addresses contain commas). Columns of every source file:
`entity_id, business_name, business_address, country`.

## 3. Sources
* **Source 1** - clean, deduplicated reference (2000 businesses in train). Full legal suffixes, canonical
  address `12 MG Road, Chennai, Tamil Nadu 600001`, full country name, no missing fields. 16 business categories, 8 countries
  (India/US dominant; Germany, France, Singapore rare). It deliberately contains **chain branches** (same name, different address)
  and **look-alike siblings** (similar name in the same city, or same street with a different business) - these are separate entities.
* **Source 2** - noisy; skewed toward *abbreviations, legal-suffix changes, typos/misspellings, punctuation/case*.
* **Source 3** - noisy in a *different* way; skewed toward *missing address components, landmark-based addresses, address
  re-ordering, shortened names, transliteration, missing PIN codes*.
* Both S2 and S3 also contain **orphan records** (businesses that do not exist in Source 1) that are built to be confusable
  with Source 1 businesses (see section 7).
* IDs (`S1-00001`, `S2-00001`, ...) are assigned after a random shuffle, so numbering and row order carry no information.

## 4. Ground truth
`train_ground_truth.tsv` has `source1_entity_id, matched_entity_ids` - one row per S1 id, matches comma-separated (S2 ids and S3 ids only),
empty for singletons. Internally each business has a hidden `master_entity_id`; ground truth is derived from it, so it is
consistent by construction. The hidden id never appears in the source files (only in `evaluation/audit_*.tsv`).

## 5. Noise generation
Each S2/S3 record draws one *noise profile* (weights differ per source and differ again in the test split):
`exact/near-exact, abbreviation, spelling/typo, address variation, missing field, word-order/punctuation, complex (3-4 stacked ops)`.
Name ops: legal-suffix variants (Private Limited/Pvt Ltd/Pvt. Ltd.), Corp/Co/Inc/&/and, tech/intl/engg-style abbreviations,
initials (A.B.C), dropped/shortened words, keyboard typos and common misspellings, transliteration (Lakshmi/Laxmi, v/w, ee/i), accent stripping,
word re-ordering, punctuation/case/spacing. Address ops: Rd/St/Ave/Apt/Bldg abbreviations, `No. 12` / `#12` / `12-A`, dotted `M.G.`,
missing PIN/state/city/house-number/unit, landmark replacement (`Near Gandhi Mall`), component re-ordering, alternative separators, PIN formats, typos.
Country: canonical name, variants (IN, USA, U.K., Deutschland ...) or missing. Some records get several problems at once.
**The test split uses different profile weights, different countries mix and extra unseen ops** (`The` prefix, vowel-dropped words, `Group` suffix,
umlaut expansion, `PIN 600001`/`ZIP` prefixes, unit-first addresses, country appended to the address).

## 6. Positive / singleton / multi-match examples (train)
```
S1-00001 | Lakshmi Electricals Private Limited | 168B Main Salai, Ganesh Puram, Chennai, Tamil Nadu 600216 | India
   -> S2-00334 | LAKSHMI ELECTRICALS  PRIVATE LIMITED | 168B MAIN SALAI GANESH PURAM CHENNAI TAMIL NADU 600216 | India
   -> S2-00837 | Lakshmi Electricals Private Ltd | 168B Main Sl, Ganesh Puram, Chennai, Tamil Nadu 600216 | India
   -> S3-01132 | Electricals, Lakshmi Private Limited | Tamil Nadu 600216, Chennai, Ganesh Puram, 168B Main Salai | India
   -> S3-01370 | Lakshmi Electricals P Ltd | 168B Main Salai, Ganesh Puram, Chennai, Tamil Nadu 600216 | India
   -> S3-02013 | Lakshmi Electricals Private Limited |  | India

S1-00002 | Riansolon Softworks Inc. | 8 Bloor Drive, Building E, Calgary, AB T3A 2J7 | Canada
   -> S2-01294 | Riansolon Softworks Inc | 8 Bloor Drive, Building E, Calgary, AB T3A 2J7 | CAN
   -> S2-01370 | Riansolon Softworjs Incorporatted | 8 Bloor Drive, Building E, Calgary, AB T3A 2J7 | Canada
   -> S3-00523 | Riansolon Softworks Inc. | 8 Bloor Drive, Building E, Calgary, AB T3A 2J7 | <missing>
   -> S3-00653 | Riansolon Softworks Inc. | 8 Bloor Drive, Building E, Calgary, AB T3A 2J7 | <missing>

S1-00004 | Zengensys Cloud Private Limited | 17 MG Cross Street, Mahatma Vihar, New Delhi, Delhi 110572 | India
   -> S2-00369 | Zengensys Cloud Private Limited | 17 MG Cross Street, Mahatma Vihar, New Delhi | India
   -> S2-00658 | Zengensys Cloud Pvt Limited | 17 MG Cross St, Mahatma Vihar, New Delhi, Delhi 110572 | India
   -> S3-01224 | Zengensys Pvt Ltd | 17 MG Cross St., Mahatma Vihar, New Delhi, Delhi | India
   -> S3-01248 | zengensys cloud private lmited | 17 MG Cross Street Mahatma Vihar New Delhi Delhi | Bharat
S1-00010 | Prime Chola Multispeciality Hospital Limited | 158 Ring Cross Street, Saraswathi Puram, Pune, Maharashtra 411626 | India   -> (no matches)
S1-00013 | Devi Multispeciality Hospital | 16A Residency Main Road, Block E, Vasant Layout, Bengaluru, Karnataka 560758 | India   -> (no matches)
S1-00027 | Kalyani Textiles Private Limited | 165A Subramaniam Lane, Block B, Saraswathi Park, Madurai, Tamil Nadu 625154 | India   -> (no matches)
```

## 7. Negative examples (train)
Hard negatives are of five kinds: same name/different address, similar name/different address, same city/different business,
similar address/different name, and same country/unrelated. They come from (a) S1 chain branches and siblings, and (b) orphan decoy
records in S2/S3. Records that share a name but belong to different businesses:
```
S1-00012 | Elpri Gadgets | 3269 Lake Boulevard, Atlanta, GA 30322 | United States
   vs S1-01243 | Elpri Gadgets | 8111 Cedar Lane, Suite 611, Phoenix, AZ 85060 | United States   [DIFFERENT businesses: same name, different address]
S1-00024 | Summit Pharmacy LLC | 9242 Washington Street, Suite 824, San Jose, CA 95186 | United States
   vs S1-00422 | Summit Pharmacy LLC | 6308 42nd Street, San Jose, CA 95126 | United States   [DIFFERENT businesses: same name, different address]
S2-00192 | Sri Krishna Cloud Limited | 157 Subramaniam Main Road, Indira Puram, Coimbatore, Tamil Nadu 641920 | Bharat   [NOT a match for any S1 record]
S2-00233 | Meto Fairview | 61 Mill Avenue, Behind Victoria Park, Birmingham | United Kingdom   [NOT a match for any S1 record]
S2-00013 | Vevec Gadgets Pty Ltd | 135 Bourke Avenue, Sydney NSW 2024 | Australia   [NOT a match for any S1 record]
S2-00081 | Axmonlo | Hauptallee 104a, 60018 Frankfurt | Germany   [NOT a match for any S1 record]
S2-00141 | ALPHA DUPONT MULTISPECIALITY HOSPITAL | 282 Place Victor Hugo, Marseille | <missing>   [NOT a match for any S1 record]
S2-00135 | Murguan Analytics Private Limited |  | India   [NOT a match for any S1 record]
S2-00416 | Cooper & Bennetu Apparwl | 5227 Congress Street, Miami, FL 33155 | United States   [NOT a match for any S1 record]
S2-00517 | Gokul Plus Medicals | 194B Subramaniam Sl, Shivaji Extension, Hyderabad, Telangana 500938 | India   [NOT a match for any S1 record]
```

## 8. Load with pandas
```python
import pandas as pd
kw = dict(sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")   # keep_default_na=False keeps '' for missing fields
s1 = pd.read_csv("dataset/train/train_source1.tsv", **kw)
s2 = pd.read_csv("dataset/train/train_source2.tsv", **kw)
s3 = pd.read_csv("dataset/train/train_source3.tsv", **kw)
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", **kw)
gt["matched"] = gt.matched_entity_ids.apply(lambda x: x.split(",") if x else [])
pairs = pd.read_csv("ml/training_pairs.tsv", sep="\t")   # blank features become NaN
```

## 9. `training_pairs.tsv`
Columns: `source1_entity_id, candidate_entity_id, candidate_source` + 14 similarity features + `label` (1 = same business).
* **Positives:** every true (S1, S2/S3) pair from the ground truth.
* **Negatives, per S1 entity:** a cheap *blocking* step (inverted index on rare name tokens, weight 3, and rare address tokens, weight 1) scores all
  S2/S3 records; the 3 highest-scoring non-matches are taken as hard negatives, 2 random non-matches from the remaining block as medium negatives,
  and 1 random record from the whole pool as an easy negative. Blocking uses record text only - **never the label**.
* **Features** are computed only from the two records' text (normalised: lower-case, accent-stripped, `&`->`and`, punctuation removed) with a pure-Python
  LCS (indel) ratio: `*_ratio`, `*_partial_ratio` (best window), `name_token_ratio` (token-sort), Jaccard, exact flags, length differences (characters),
  shared-token counts. Scores are 0-100. `country_match` = 1 same canonical country, 0 different, **-1 unknown** (either side missing).
  If either address is missing, the address similarity features are blank (NaN) and `address_exact=0`.
* Label is looked up from the ground truth only *after* features are computed.

## 10. Statistics (train)
```
{
 "records": {
  "S1": 2000,
  "S2": 3001,
  "S3": 3000
 },
 "unique_underlying_businesses_S1": 2000,
 "unique_underlying_businesses_incl_decoys": 2258,
 "positive_matches": 5671,
 "positives_S2": 2851,
 "positives_S3": 2820,
 "singleton_s1": 300,
 "single_match_s1": 173,
 "multi_match_s1": 1527,
 "max_matches_one_s1": 8,
 "cross_source_multi": 1301,
 "orphan_records": {
  "S2": 150,
  "S3": 180
 },
 "decoy_kinds": {
  "decoy_same_name_diff_addr": 121,
  "decoy_same_city_diff_biz": 51,
  "decoy_similar_name_diff_addr": 82,
  "decoy_similar_addr_diff_name": 54,
  "decoy_random": 22
 },
 "s1_origins": {
  "base": 1740,
  "sibling_similar_address": 52,
  "chain_branch": 150,
  "sibling_similar_name": 58
 },
 "s1_countries": {
  "India": 780,
  "Canada": 150,
  "Germany": 85,
  "France": 105,
  "United States": 533,
  "United Kingdom": 139,
  "Australia": 135,
  "Singapore": 73
 },
 "missing": {
  "S2": {
   "address_missing_pct": 3.9,
   "country_missing_pct": 5.3,
   "pin_missing_pct": 13.46
  },
  "S3": {
   "address_missing_pct": 7.5,
   "country_missing_pct": 9.2,
   "pin_missing_pct": 34.9
  },
  "S1": {
   "address_missing_pct": 0.0,
   "country_missing_pct": 0.0,
   "pin_missing_pct": 0.0
  }
 },
 "noise_profile": {
  "S2": {
   "abbrev": 35.0,
   "exact": 20.1,
   "typo": 19.0,
   "order": 9.2,
   "address": 6.7,
   "missing": 6.4,
   "complex": 3.5
  },
  "S3": {
   "address": 25.9,
   "exact": 19.7,
   "abbrev": 14.5,
   "order": 13.3,
   "missing": 12.8,
   "typo": 7.4,
   "complex": 6.4
  }
 },
 "noise_profile_overall": {
  "abbrev": 24.7,
  "exact": 19.9,
  "address": 16.3,
  "typo": 13.2,
  "order": 11.2,
  "missing": 9.6,
  "complex": 4.9
 },
 "name_op_freq_pct": {
  "S2": {
   "abbrev": 30.3,
   "legal_variant": 28.3,
   "typo": 20.2,
   "case": 12.4,
   "punct": 9.1,
   "initials": 8.0,
   "shorten": 5.5,
   "spaces": 4.8,
   "reorder": 3.1,
   "legal_drop": 1.4,
   "drop_word": 1.2,
   "translit": 0.8
  },
  "S3": {
   "case": 19.7,
   "legal_variant": 15.2,
   "abbrev": 12.6,
   "punct": 12.1,
   "shorten": 10.8,
   "reorder": 10.3,
   "typo": 10.1,
   "translit": 4.8,
   "legal_drop": 4.7,
   "spaces": 4.0,
   "initials": 3.7,
   "drop_word": 2.4
  }
 },
 "addr_op_freq_pct": {
  "S2": {
   "abbr": 28.8,
   "state_abbr": 11.4,
   "drop_pin": 9.3,
   "punct": 8.8,
   "typo": 6.1,
   "sep": 4.7,
   "missing_addr": 3.9,
   "drop_unit": 3.0,
   "numstyle": 2.1,
   "upper": 1.9,
   "drop_state": 1.7,
   "dotted": 1.5,
   "drop_street": 1.5,
   "spaces": 1.2,
   "landmark": 1.1,
   "reorder": 0.8,
   "pinfmt": 0.5,
   "drop_bno": 0.4,
   "drop_city": 0.4,
   "accent": 0.2
  },
  "S3": {
   "drop_pin": 25.1,
   "abbr": 17.1,
   "reorder": 11.3,
   "landmark": 10.8,
   "drop_unit": 9.7,
   "drop_state": 9.4,
   "punct": 8.1,
   "missing_addr": 7.5,
   "drop_bno": 5.7,
   "state_abbr": 5.0,
   "sep": 4.7,
   "pinfmt": 4.5,
   "typo": 3.5,
   "drop_street": 3.4,
   "drop_city": 3.3,
   "spaces": 2.9,
   "upper": 2.8,
   "dotted": 0.9,
   "numstyle": 0.8,
   "accent": 0.6
  }
 },
 "country_variant_pct": {
  "S2": 15.0,
  "S3": 25.2
 },
 "pairs_total": 17671,
 "pairs_pos": 5671,
 "pairs_neg": 12000,
 "pairs_hard_neg": 1287,
 "pairs_same_name_diff_biz": 510,
 "pairs_same_addr_diff_biz": 0,
 "pairs_by_source": {
  "S2": 8881,
  "S3": 8790
 }
}
```
Exact-string baselines on the full S1 x (S2 u S3) space (train):
```
{
 "exact_name": {
  "predicted": 3824,
  "true_positive": 3196,
  "precision": 0.836,
  "recall": 0.564
 },
 "exact_name_and_address": {
  "predicted": 1431,
  "true_positive": 1431,
  "precision": 1.0,
  "recall": 0.252
 }
}
```
Test split (600 S1 / ~900 / ~900) statistics:
```
{
 "records": {
  "S1": 600,
  "S2": 900,
  "S3": 900
 },
 "positive_matches": 1609,
 "singleton_s1": 108,
 "multi_match_s1": 434,
 "noise_profile_overall": {
  "abbrev": 19.8,
  "address": 18.4,
  "exact": 13.5,
  "missing": 12.8,
  "typo": 12.5,
  "order": 11.7,
  "complex": 11.3
 },
 "decoy_kinds": {
  "decoy_same_name_diff_addr": 63,
  "decoy_random": 19,
  "decoy_similar_addr_diff_name": 33,
  "decoy_same_city_diff_biz": 21,
  "decoy_similar_name_diff_addr": 55
 },
 "s1_countries": {
  "Singapore": 61,
  "United States": 133,
  "United Kingdom": 70,
  "Germany": 64,
  "Canada": 49,
  "Australia": 38,
  "India": 123,
  "France": 62
 }
}
```
Validation checks (all must be PASS): train 20/20, test 20/20.

## 11. Suggested evaluation
Split **by S1 entity** (GroupKFold on `source1_entity_id`), never by pair, so near-duplicate records of one business cannot leak across folds.
Judge the whole pipeline, not only the classifier: blocking recall, pair-level precision/recall/F1 and PR-AUC, then the final
one-to-many assignment. Tune the threshold on out-of-fold predictions; keep the test split untouched until the end and score with
`evaluation/test_ground_truth.tsv`. Report per-source (S2 vs S3), per-country and per-noise-profile (via `audit_*.tsv`) metrics.
