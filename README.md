# 🧬 Auto Neural Network Designer
## Neural Architecture Search via Genetic Algorithms

A mini AutoML system that automatically discovers the best neural network
architecture for your dataset using evolutionary computation.

---

## 📁 Folder Structure

```
auto_nas/
│
├── app.py                    ← Main Streamlit application (entry point)
│
├── core/
│   ├── __init__.py
│   ├── genetic_algorithm.py  ← GA logic: init, selection, crossover, mutation
│   └── model_builder.py      ← Keras model builder + fitness function
│
├── utils/
│   ├── __init__.py
│   ├── preprocessor.py       ← Data cleaning, encoding, scaling, splitting
│   └── plotter.py            ← Matplotlib charts for fitness & architecture
│
├── sample_dataset.csv        ← Demo dataset (employee promotion prediction)
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## 🚀 How to Run

### 1. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 🧠 How It Works

```
Upload CSV → Preprocess → Initialize Population → [Evaluate → Select → Crossover → Mutate] × N generations → Best Model
```

1. **Upload** your CSV and select the target column
2. **Preprocessing** handles missing values, encodes categories, scales features
3. **Genetic Algorithm** evolves a population of neural network architectures:
   - Each genome = list of hidden layer sizes e.g. `[128, 64, 32]`
   - Fitness = validation accuracy (or 1/(1+loss)) - complexity penalty
   - Top-K elites survive each generation
   - Children created by crossover + mutation
4. **Best architecture** is trained fully and saved

---

## 🎛️ GA Parameters Explained

| Parameter       | What it does                                      |
|-----------------|---------------------------------------------------|
| Population Size | Number of architectures per generation            |
| Generations     | How many evolution cycles to run                  |
| Top-K           | How many best models survive each generation      |
| Mutation Rate   | Probability of random architecture change         |
| Epochs/Model    | Training epochs per model (lower = faster search) |

---

## 📊 Sample Dataset

`sample_dataset.csv` contains employee data with features:
- age, salary, experience, department, performance_score
- Target: `promoted` (0 or 1) — binary classification

---

## 🔧 Advanced Improvements You Can Add

1. **Early stopping** — Stop training if val_loss doesn't improve
2. **Hyperparameter evolution** — Evolve learning rate, dropout, batch size too
3. **NSGA-II multi-objective** — Optimize accuracy AND speed simultaneously
4. **Bayesian optimization** — Replace random mutation with smarter search
5. **Transfer learning** — Start from pre-trained weights
6. **NAS with weight sharing** — One supernetwork, sample subnets (DARTS)
7. **Distributed evaluation** — Run models in parallel on multiple cores
8. **Export to ONNX** — Deploy the best model cross-platform
9. **Automated feature selection** — Evolve which features to use too
10. **Leaderboard persistence** — Save all runs to SQLite for comparison

---

## 🎓 Academic Relevance

This project implements:
- **Neural Architecture Search (NAS)** — Active research area at Google, Meta, Microsoft
- **Evolutionary Computation / Soft Computing** — Genetic Algorithms
- **AutoML** — Automated Machine Learning pipeline
- **Keras Functional API** — Dynamic model construction

Suitable for: Final Year Project, Soft Computing course, AI/ML portfolio

---

## 📦 Dependencies

- `streamlit` — Web UI
- `tensorflow` — Neural network training
- `scikit-learn` — Preprocessing & splitting
- `pandas` / `numpy` — Data handling
- `matplotlib` — Visualization
