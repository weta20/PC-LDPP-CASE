# Benchmark Dataset for Process-Constrained Laser Drilling (PC-LDPP)

## 1. Overview
This repository contains a collection of large-scale benchmark datasets for the **Laser Drilling Path Planning (LDPP)** problem in semiconductor manufacturing. 

The dataset is designed to challenge Traveling Salesman Problem (TSP) solvers by introducing **Process-Specific Constraints**, where the cost of movement depends not only on distance but also on the kinetic stability of the laser head across a **four-point sequence (Quad-Sequence)**.

## 2. Problem Context
In high-density chip manufacturing, the goal is to traverse $10^5$ to $10^6$ coordinates. To maintain precision, the path must minimize frequent sharp turns and sudden accelerations. 

Researchers using this data are encouraged to apply a **Look-Up Table (LUT)** cost model:
$$\text{Total Cost} = \alpha \cdot \text{Distance} + \beta \cdot \text{Turning Penalty}(\Phi)$$
Where $\Phi$ is determined by the geometric relationship of four consecutive points ($p_a \to p_b \to p_c \to p_d$).

## 3. Data Specification

### File Format
The data is provided in `.csv` or `.tsp` (TSPLIB compatible) formats. Each file represents a specific chip layout.

### Column Descriptions
| Column | Name | Description | Unit |
| :--- | :--- | :--- | :--- |
| 1 | `ID` | Unique index of the hole | Integer |
| 2 | `X` | X-coordinate of the hole center | Micrometers ($\mu m$) |
| 3 | `Y` | Y-coordinate of the hole center | Micrometers ($\mu m$) |
| 4 | `Type` | Cluster/Layer category (if applicable) | Category ID |

### Physical Parameters
- **Coordinate Scale:** Usually $0$ to $200,000 \mu m$ (Full Wafer/Panel scale).
- **Hole Diameter:** Typically $10 \sim 50 \mu m$.
- **Precision Requirement:** Sub-micron repeatability.

## 4. Quad-Sequence Cost Interpretation
When evaluating a path sequence $( \dots, p_a, p_b, p_c, p_d, \dots )$, the "Turning Penalty" should be applied at the jump segment $p_b \to p_c$.



- **Segment $ab$:** Active drilling (Laser ON).
- **Segment $bc$:** Positioning jump (Laser OFF).
- **Segment $cd$:** Next active drilling (Laser ON).
- **Penalty Logic:** If $\vec{v}_{ab}$ and $\vec{v}_{cd}$ are not collinear, a penalty value $\Phi$ from the provided `cost_table.csv` should be added to the objective function.

## 5. Dataset Categories
1. **Regular Arrays (`/data/regular/`):** Holes arranged in strict grids. Tests the solver's ability to find "S-curves" or "Z-curves".
2. **Stochastic Distributions (`/data/random/`):** Irregularly placed holes. Tests the global optimization capability.
3. **Hybrid Chips (`/data/hybrid/`):** Large-scale instances ($n > 500,000$) combining dense clusters and sparse regions.

## 6. Comparison Protocol
To ensure fair comparison in your publications, please report:
1. **Total Euclidean Distance** (Standard TSP metric).
2. **Total Turning Count** (Number of times $\theta_{abcd} \neq 0$).
3. **Weighted Objective Value** using $\alpha=1, \beta=100$ (as used in the original paper).

## 7. Citation
If you use these benchmark instances in your research, please cite:

```bibtex
@dataset{YourName2026,
  author = {Your Name},
  title = {A Large-Scale Quad-Sequence Benchmark for Laser Drilling Path Planning},
  year = {2026},
  publisher = {GitHub},
  journal = {[https://github.com/YourUsername/Chip-Laser-Drilling-Dataset](https://github.com/YourUsername/Chip-Laser-Drilling-Dataset)}
}
