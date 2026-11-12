<div align="center">
<h1>FastPGM: Fast Probabilistic Graphical Model Learning and Inference</h1>
</div>

![](docs/source/fig/doc_cover.jpg)

[**Documentation**](https://fastpgm.readthedocs.io/en/latest/index.html) 
| [**Parameters**](https://fastpgm.readthedocs.io/en/latest/param_toc.html) 
| [**Knowledge base**](https://fastpgm.readthedocs.io/en/latest/basic_toc.html)
| [**Python interface**](https://fastpgm.readthedocs.io/en/latest/python_toc.html)

**Fast-PGM** is an open-source C++ library that aims to help practitioners easily and efficiently 
apply probabilistic graphical models (PGMs) to solve real-world problems. FastBN exploits multi-core CPUs 
to achieve high efficiency. Key features of FastBN are as follows:
- Wide coverage of different tasks and algorithms related to PGMs, including structure learning 
(PC-stable), parameter learning (maximum likelihood estimation), exact inference (junction tree, 
variable elimination, brute force), and approximate inference (loopy belief propagation, 
probabilistic logic sampling, likelihood weighting, self-importance sampling, AIS-BN, EPIS-BN).
- Support classification, through the building blocks of structure learning, parameter learning 
and inference.
- Support Python interfaces.
- Support BN sample generation, dataset and network format convertor, etc.


## Installation

### 1. Prerequisites
- **C++ Compiler**: GCC (>= 9.0) or Clang supporting C++17.
- **Build Tools**: CMake (>= 3.10), Make.
- **OpenMP**: Required for multi-threaded parallel execution.

### 2. Clone Repository & Submodules
> **Note:** The libraries in `lib/` (GoogleTest, TinyXML2, ARFF, Stats, GCEM) and `python/pybind11` are managed as **Git Submodules**. Be sure to clone recursively.

```bash
# Clone with all submodules
git clone --recursive <repo-url>
cd FastPGM
```

If you or your collaborator already cloned the repository without `--recursive` (resulting in empty `lib/` directories), pull the submodules by running:
```bash
git submodule update --init --recursive
```

### 3. Build FastPGM
```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```
Upon successful compilation, the executable `fastpgm` will be generated in the `build/` directory.

---

## Quick Start

All commands should be executed from the `build/` directory (or specify `-f` paths relative to `dataset/`).

### 1. Default Run
By default, FastPGM runs structure learning with PC-Stable using 1 thread on 5,000 samples from the ALARM network:
```bash
cd build
./fastpgm
```

### 2. Bayesian Network Structure Learning (Job 0: PC-Stable)
Learn network structure from samples and evaluate against reference ground-truth `.bif`:
```bash
# Run with 4 threads, CI-level parallelism, group size = 1
./fastpgm -j 0 -m 0 -t 4 -g 1 -a 0.05 -f1 alarm/alarm.bif -f2 alarm/alarm_s5000
```

#### Parallelism Granularities (`-p`):
FastPGM supports 3 granularities of parallelism (referenced in IPDPS 2022 paper, Fig. 2):
- `-p 0`: **CI-level parallelism** (Default Fast-BNS with dynamic work pool)
- `-p 1`: **Edge-level parallelism** (Parallelizes across edges in each depth)
- `-p 2`: **Sample-level parallelism** (Parallelizes across samples inside CI test table counting)

Example:
```bash
# Edge-level parallelism with 8 threads
./fastpgm -j 0 -m 0 -t 8 -p 1 -f1 insurance/insurance.bif -f2 insurance/insurance_s5000
```

### 3. Structure Learning + Parameter Learning (Job 1)
Learn structure (PC-Stable) followed by Maximum Likelihood Estimation (MLE):
```bash
./fastpgm -j 1 -m 0 -t 4 -f1 alarm/alarm.bif -f2 alarm/alarm_s5000
```

### 4. Exact Inference (Job 2: Junction Tree)
Perform exact inference on evidence and evaluate potential tables:
```bash
./fastpgm -j 2 -m 1 -t 4 -f0 alarm/alarm.xml -f3 alarm/testing_alarm_1k_p20_csv -f4 alarm/alarm_1k_pt
```

---

## Detailed Guides & Documentation

- [STRUCTURE_AND_OPERATION.md](docs/guidance/STRUCTURE_AND_OPERATION.md): Detailed project structure, command options, and output interpretation.
- [ADDING_DATASET.md](docs/guidance/ADDING_DATASET.md): Guide to downloading, generating samples, and experimenting with new datasets (e.g., Hepar2).
- [FASTBN_ALG.md](docs/guidance/FASTBN_ALG.md): Complete pseudocode and source-code mapping for Fast-BNS.
- [CHANGING_HISTORY.md](docs/guidance/CHANGING_HISTORY.md): Modification history of C/C++ source code.
- [Python & R Notebooks](notebooks/): Jupyter (`reproduce_v1.ipynb`) and R Markdown (`R_notebooks/bn_learn.Rmd`, `pcalg_learn.Rmd`) reproduction experiments.


## Citation

If you find FastPGM useful, please consider citing our papers.

Fast-BNS in IPDPS 2022:
```
@inproceedings{jiang2022fast,
  title={Fast Parallel Bayesian Network Structure Learning},
  author={Jiang, Jiantong and Wen, Zeyi and Mian, Ajmal},
  booktitle={2022 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  pages={617--627},
  year={2022}
}
```

Fast-BNS-v2 in TPDS (2024):
```
@article{yang2024parallel,
  title={Parallel and Distributed Bayesian Network Structure Learning},
  author={Yang, Jian and Jiang, Jiantong and Wen, Zeyi and Mian, Ajmal},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  year={2024},
  volume={35},
  number={4},
  pages={517-530}
}
```

Fast-BNI in PPoPP 2023:
```
@inproceedings{jiang2023fast,
  title={Fast Parallel Exact Inference on Bayesian Networks},
  author={Jiang, Jiantong and Wen, Zeyi and Mansoor, Atif and Mian, Ajmal},
  booktitle={Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
  pages={425--426},
  year={2023}
}
```

Faster-BNI in TPDS (2024):
```
@article{jiang2024faster,
  title={Faster-BNI: Fast Parallel Exact Inference on Bayesian Networks}, 
  author={Jiang, Jiantong and Wen, Zeyi and Mansoor, Atif and Mian, Ajmal},
  journal={IEEE Transactions on Parallel and Distributed Systems},
  year={2024},
  volume={35},
  number={8},
  pages={1444-1455}
```

Fast-PGM (inference only) in USENIX ATC 2024:
```
@inproceedings{jiang2024fast,
  title={Fast inference for probabilistic graphical models},
  author={Jiang, Jiantong and Wen, Zeyi and Mansoor, Atif and Mian, Ajmal},
  booktitle={2024 USENIX Annual Technical Conference (USENIX ATC 24)},
  pages={95--110},
  year={2024}
}
```

Fast-PGM (the library): 
```
@article{jiang2024fastpgm,
  title={Fast-PGM: Fast Probabilistic Graphical Model Learning and Inference},
  author={Jiang, Jiantong and Wen, Zeyi and Yang, Peiyu and Mansoor, Atif and Mian, Ajmal},
  journal={arXiv preprint arXiv:2405.15605},
  year={2024}
}
```


## Dependency
 * [GoogleTest](https://github.com/google/googletest)
 * [TinyXML2](https://github.com/leethomason/tinyxml2)
 * [ARFF](https://github.com/LinjianLi/ARFF)
 * [StasLib](https://github.com/jjiantong/stats)
 * [GCE-Math](https://github.com/kthohr/gcem)
 * [pybind11](https://github.com/pybind/pybind11)


