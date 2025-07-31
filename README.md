# Farcaster Social Graph

This project develops a sybil detection system for the Farcaster social network by combining two approaches: the SybilSCAR social graph algorithm and a custom machine learning model. By analyzing both network topology and user behavior data obtained through Neynar's API, our solution provides accurate probability scores identifying potential sybil accounts.

A public API is available for developers to integrate these detection capabilities into their own applications. Additionally, we've implemented a Farcaster Frame allowing users to interact with the detection system directly within the Farcaster ecosystem. For implementation details, code examples, and documentation, explore [this repository](https://github.com/bleu/op-report-sybil-farcaster-frame).

## Model documentation

#### Data used

These are the data sources used at this project:

- **Neynar**: This is our main source of data, providing informations such as users casts, likes and follower relations. For a more detailed data description, check [Farcaster data from Neynar](https://docs.dune.com/data-catalog/community/farcaster/overview).
- **Bot or Not**: To create a list of sybils, we considered [Bot or Not](https://warpcast.com/botornot) bot outputs from the Neynar casts data. It was possible to create a base bot label list with approximately 4,000 lines, which is appended to `labels.csv`.
- **Human inspection label**: To get non-sybil labels, we manually inspected users filtered by some conditions (like ENS names and high Gitcoin scores), particularly focusing on examining the accounts they followed. This manual verification process resulted in a validated list of 416 human accounts, available at `labels.csv` in this repo

#### Overview

The model takes Farcaster users data as input to return their proabiblity (0<= p <= 1) of being a sybil. The final output is the average of two model results: Sybilscar and a Machine Learning model. Not always it is possible to compute Sybilscar or Machine Learning model outputs, so if one of the models result is missing, the other one is used instead. There is also the possibility of having both null values, which will lead to an unknown result.

#### Sybilscar

[SybilSCAR](https://www.researchgate.net/publication/317506206_SybilSCAR_Sybil_Detection_in_Online_Social_Networks_via_Local_Rule_based_Propagation) is an algorithm designed to detect fake accounts (Sybils) in social networks based on the links between users (on our case, the mutual follow interactions).

- It formulates Sybil detection as a local rule-based framework where nodes iteratively exchange messages with their neighbors
- Incorporates both node features and graph structure through regularization
- Uses a probabilistic model where each node has a probability of being legitimate or Sybil

A crucial implementation consideration emerged during our analysis: SybilSCAR requires calculating Sybil probabilities for the entire network simultaneously, rather than processing individual nodes in isolation. This architectural constraint necessitates a batch processing approach.

#### Machine Learning model

Three models are ensembled to obtain the final Machine Learning model output - XGBoost, Random Forest, and LightGBM - using over 100 features extracted from the Neynar dataset. The feature set encompasses three main categories:

- User identity metrics, which capture account-specific characteristics and verification status
- Network analysis metrics, which examine user interactions and connection patterns
- Temporal behavior metrics, which track posting frequency and activity patterns over time

#### Results

The main success metric observed in this project is the ROC AUC score. Here is a summary of the metrics obtained in the labeled data:

| Model            | ROC AUC score |
| ---------------- | ------------- |
| SybilSCAR        | 0.9545        |
| Machine Learning | 0.9899        |
| Final ensemble   | 0.9922        |

In order to inspect if the results make sense outside the labeled samples, it's possible to check the probability distribution on all users:

![ML sybil probabilities distribution](https://github.com/user-attachments/assets/c718eda6-c024-4d12-88ae-7b8f6ea072e2)

![SybilSCAR sybil probabilities distribution](https://github.com/user-attachments/assets/a079bc8c-6c53-4f01-b9a2-100f51dd1af2)

![Ensembled sybil probabilities distribution](https://github.com/user-attachments/assets/05ce86b0-44b7-410b-9ae1-12cbff93b754)

## Usage

#### Requirements

- python v3.12
- poetry

#### Download Neynar data

**Farcaster**

To check latest timestamps:

```bash
aws s3 ls s3://tf-premium-parquet/public-postgres/farcaster/v2/full/
```

To download data for a timestamp:

```bash
aws s3 cp s3://tf-premium-parquet/public-postgres/farcaster/v2/full/ ./data/raw --recursive --exclude "*" --include "*-<end_timestamp>.parquet"
```

**Nindexer**

To check latest timestamps:

```bash
aws s3 ls s3://tf-premium-parquet/public-postgres/nindexer/v3/1/full/
```

To download data for a timestamp:

```bash
aws s3 cp s3://tf-premium-parquet/public-postgres/nindexer/v3/1/full/ data/raw --recursive  --exclude "*"  --include "*-<end_timestamp>.parquet" --profile neynar_parquet_exports
```

#### Prepare enviroment

Go to notebooks:
```bash
cd notebooks
```

Install dependencies:

```bash
cd farcaster-social-graph-api
poetry install
```

Activate environment:
```bash
poetry shell
python -m ipykernel install --user --name farcaster_social_graph
```

#### Notebooks overview

**Bot or Not data extraction and evaluation**:
- `notebooks/01-merge-data.ipynb`
- `notebooks/02-extract-bot-or-not-checks.ipynb`
- `notebooks/10-evaluate-bot-or-not-quality.ipynb`

**Benign labeling process**:
- `notebooks/11-test-verifications-to-get-human-labels.ipynb`
- `notebooks/12-manual-human-labeling.ipynb`

**SybilSCAR test**:
- `notebooks/13-inspect-sybilscar-result.ipynb`

**Machine learning model development and tests**:
- `notebooks/14-explore-features.ipynb`
- `notebooks/15-consolidate-features.ipynb`
- `notebooks/16-add-new-features.ipynb`
- `notebooks/17-apply-self-training.ipynb`

**Final model tests**:
- `notebooks/22-get-auc-of-ensembled-test.ipynb`
- `notebooks/23-inspect-model-outputs.ipynb`

#### Test the algorithm

The main routine can be called using:
```bash
poetry run python farcaster_social_graph_api/jobs.py
```
It may take some minutes to run.


## Useful links:

[Project proposal on charmverse](https://app.charmverse.io/op-grants/farcaster-social-graph-13991200096869738)
[Project documentation](https://www.notion.so/bleu-builders/Farcaster-Social-Graph-Documentation-13bc4ae7dcbd80418174f7312354cfea?source=copy_link)
[Report Sybil - Farcaster frame](https://github.com/bleu/op-report-sybil-farcaster-frame)



