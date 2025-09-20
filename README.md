# dvcpipeline-NitishSingh

This repository demonstrates a data version control (DVC) pipeline using Python. It aims to showcase best practices for managing data science workflows, tracking data and model versions, and ensuring reproducibility in machine learning projects.

## Features

- **DVC Integration**: Easily track datasets, models, and pipeline stages.
- **Python Scripts**: Modular scripts for data loading, preprocessing, training, and evaluation.
- **Reproducibility**: All steps in the workflow are versioned and can be reproduced by anyone.
- **Collaborative Workflow**: Supports collaboration among data scientists with efficient data sharing and experiment tracking.

## Folder Structure

```
.
├── data/               # Raw and processed data tracked by DVC
├── src/                # Python source code for pipeline stages
├── models/             # Model files tracked by DVC
├── dvc.yaml            # DVC pipeline definition
├── dvc.lock            # DVC pipeline lock file

```

## Getting Started

### Prerequisites

- [Python 3.7+](https://www.python.org/downloads/)
- [DVC](https://dvc.org/doc/install)
- [Git](https://git-scm.com/)

Install dependencies:

```bash
pip install -r requirements.txt
```

Install DVC (if not already installed):

```bash
pip install dvc
```

### Cloning and Setting Up

1. Clone the repository:

    ```bash
    git clone https://github.com/srvmann/dvcpipeline-NitishSingh.git
    cd dvcpipeline-NitishSingh
    ```

2. Pull data and models managed by DVC:

    ```bash
    dvc pull
    ```

### Running the Pipeline

To run the complete pipeline as defined in `dvc.yaml`:

```bash
dvc repro
```

You can also run individual stages using `dvc repro <stage-name>`.

## Usage

Modify Python scripts in the `src/` directory to customise data processing and model training. Use DVC commands to add new data, track experiments, and share results.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This repository is licensed under the MIT License.

## References

- [DVC Documentation](https://dvc.org/doc)
- [DVC Tutorials](https://dvc.org/doc/tutorials)
