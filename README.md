# PC-LDPP: Process-Constrained Laser Drilling Path Planning

## 1. Overview
**PC-LDPP** is an open-source framework designed to solve the path planning problem for large-scale chip laser drilling. In modern semiconductor manufacturing, a single chip may contain millions of micro-holes. Minimizing the processing time while maintaining high precision is a typical **Large-Scale Traveling Salesman Problem (TSP)** with complex industrial constraints.

Unlike conventional TSP, this project introduces a **Quad-Sequence Cost Model** that accounts for the laser head's acceleration, deceleration, and turning stability through a **Look-Up Table (LUT)** approach.

## 2. Key Features
- **Large-Scale Solver:** Optimized for $10^5$ to $10^6$ hole positions.
- **Quad-Sequence Cost Model:** Evaluates the transition between two consecutive drilling segments ($a \to b$ and $c \to d$) to minimize vibration and mechanical wear.
- **LUT Optimization:** Replaces complex kinetic calculations with $O(1)$ table lookups for turning penalties.
- **Pattern Recognition:** Leverages the geometric regularity of chip hole arrays to improve solving efficiency.

## 3. Mathematical Model

The problem is formulated as a high-order TSP where the objective is to minimize the weighted sum of total distance and turning penalties.

### Objective Function
$$\min Z = w_1 \sum_{i=1}^{n-1} d_{\pi_i, \pi_{i+1}} + w_2 \sum_{i=1}^{n-3} \Phi(\pi_i, \pi_{i+1}, \pi_{i+2}, \pi_{i+3})$$

Where:
- $d_{ij}$: Euclidean distance between hole $i$ and $j$.
- $\Phi(a,b,c,d)$: Turning cost for the sequence $a \to b \to c \to d$, where $ab$ and $cd$ are processing segments, and $bc$ is a jump segment.
- $w_1, w_2$: Weighting factors for distance and process stability.

## 4. Getting Started

### Prerequisites
- Python 3.8+ or C++17 (for core solver)
- Dependencies: `numpy`, `scipy`, `matplotlib` (for visualization)

### Installation
```bash
git clone [https://github.com/weta20/PC-LDPP-CASE.git](https://github.com/weta20/PC-LDPP-CASE.git)
cd PC-LDPP
pip install -r requirements.txt
