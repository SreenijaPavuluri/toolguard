# Datasheet

Every source below is public and non-gated on the Hugging Face Hub (or ships inside an installed Python package). Nothing requires a login or license acceptance. Generated data is not committed (it is about 60 MB). Rebuild it with `python -m toolguard.data`.

| Role | Source | License | How it is used |
|---|---|---|---|
| Train attacks and benign prompts | [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Apache-2.0 | All rows, split 90/10 into train and validation |
| Train attacks and benign prompts | [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | Apache-2.0 | Prompts up to 350 words, split 90/10 into train and validation |
| Test attacks (never trained on) | [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | MIT | Human-written attacks from the Gandalf game, all splits pooled, 120 sampled for the test grid |
| Carriers, train and validation | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) `wikitext-103-raw-v1` validation split | CC BY-SA 3.0 | Real Wikipedia paragraphs |
| Carriers, test | same, `test` split (different articles) | CC BY-SA 3.0 | Real Wikipedia paragraphs |
| Carriers (markdown) and test hard negatives | README text in the metadata of installed packages whose license is MIT, BSD, Apache or PSF | per package, all permissive | Split by package name hash: two thirds train and validation, one third test |
| Test hard negatives | [databricks/databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | CC BY-SA 3.0 | Human-written answers to "how do I / steps to / tips for" questions, at least 40 words |
| Off-the-shelf detector | [protectai/deberta-v3-base-prompt-injection-v2](https://huggingface.co/protectai/deberta-v3-base-prompt-injection-v2) | Apache-2.0 | Run locally, unchanged |
| ToolGuard base model | [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 | Fine-tuned |

Skipped: `hackaprompt/hackaprompt-dataset` (gated behind terms acceptance), `aeslc` and Enron email dumps (license not clearly stated), `b-mc2/wikihow_lists` and `tatsu-lab/alpaca` (non-commercial licenses).

## Construction

* Formats: plain text, HTML page with the attack in an HTML comment, markdown README, JSON search API response with the attack in a string field.
* Depth: the inserted text is placed before paragraph `round(depth * n_paragraphs)` for depth in {0, 0.25, 0.5, 0.75, 1.0}.
* Length: target {200, 500, 1000, 2000} tokens, realised as words / 1.33. Actual MiniLM token counts per bin are in `results/dilution.json`.
* Train hard negatives: template-generated install steps, UI steps, recipes, runbooks and office emails, some with words like "ignore", "forget" and "override" used harmlessly. Inserted into carriers like attacks, labelled benign.
* Test hard negatives: documents built only from real Dolly how-to answers and real README instruction blocks (blocks that contain at least two verbs like install, run, click, set). No templates.
* Leakage audit: character 5-gram Jaccard between every test attack and every train text, and between every Dolly hard negative and every train benign prompt. Threshold 0.5. Results in `results/leakage_audit.json`.
