# Benchmark Dataset for Process-Constrained Laser Drilling (PC-LDPP)

## 1. Overview
This repository contains a collection of large-scale benchmark datasets for the **Laser Drilling Path Planning (LDPP)** problem in semiconductor manufacturing. 

The dataset is designed to challenge Traveling Salesman Problem (TSP) solvers by introducing **Process-Specific Constraints**, where the cost of movement depends not only on distance but also on the turning stability of the laser head across.

## 2. Problem Context
In high-density chip manufacturing, the goal is to traverse $0$ to $10^6$ coordinates. To maintain precision, the path must minimize frequent sharp turns and sudden accelerations. 

## 3. Data Specification

### File Format
The data is provided in `.txt`formats. Each file represents a specific chip layout.

### Column Descriptions
| Column | Name | Description | Unit |
| :--- | :--- | :--- | :--- |
| 1 | `X` | X-coordinate of the hole center | Micrometers ($\mu m$) |
| 2 | `Y` | Y-coordinate of the hole center | Micrometers ($\mu m$) |

### Physical Parameters
- **Coordinate Scale:** Usually 0to 200,000 $\mu m$ (Full Wafer/Panel scale).
- **Hole Diameter:** Typically $5 \sim 100 \mu m$.
- **Precision Requirement:** Sub-micron repeatability.



## 3. Dataset Categories
1. **Regular Cases (`/data/regular/rXX`):** Holes arranged in strict grids. Tests the solver's ability to find "S-curves" or "Z-curves".
2. **Irregular Cases (`/data/irrgular/uXX`):** Irregularly placed holes. Tests the global optimization capability.

## 6. Comparison Protocol
To ensure fair comparison in your publications, please report:
1. **Total Euclidean Distance** (Standard TSP metric).
2. **Total Turning Count** (Number of times $\theta_{abcd} \neq 0$).
3. **Weighted Objective Value** using $\alpha=1, \beta=100$ (as used in the original paper).
