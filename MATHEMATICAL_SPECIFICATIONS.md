# Mathematical Specifications

This document details every mathematical model, formula, statistical algorithm, and financial calculation implemented in the Spending Segment Analyzer.

---

## 1. Accounting and Cashflow Analytics

### 1.1 Inflow, Outflow, and Net Balance

Transactions are represented with signed values using Python `Decimal` arithmetic to avoid binary floating-point rounding errors.

- Let $T = \{t_1, t_2, \dots, t_N\}$ denote the set of all transactions.
- Each transaction $t_i$ has an amount $a_i \in \mathbb{R}$.

$$\text{Total Inflow} = \sum_{t_i \in T,\, a_i > 0} a_i$$

$$\text{Total Outflow} = \sum_{t_i \in T,\, a_i < 0} |a_i|$$

$$\text{Net Balance} = \sum_{t_i \in T} a_i = \text{Total Inflow} - \text{Total Outflow}$$

**Purpose**: Provides an exact audit of client capital without floating-point drift, establishing the base ledger balance.

### 1.2 Category Spending and Budget Variance

For a specific category $c$ and billing period defined by year $Y$ and month $M$:

$$\text{Spent}(c, Y, M) = \sum_{\substack{t_i \in T \\ t_i.\text{category} = c \\ t_i.\text{year} = Y,\, t_i.\text{month} = M \\ a_i < 0}} |a_i|$$

Given a monthly budget limit $B_c > 0$:

$$\text{Remaining}(c) = B_c - \text{Spent}(c)$$

$$\text{Utilization \%}(c) = \left( \frac{\text{Spent}(c)}{B_c} \right) \times 100$$

$$\text{OverBudget}(c) = \begin{cases} \text{True}, & \text{if } \text{Spent}(c) > B_c \\ \text{False}, & \text{otherwise} \end{cases}$$

**Purpose**: Identifies budget breaches and category leakage in real time.

### 1.3 Cumulative Spending Timeline

Let $T_{\text{exp}} = [t_{(1)}, t_{(2)}, \dots, t_{(M)}]$ be expense transactions sorted in ascending chronological order by date ($t_{(1)}.\text{date} \le t_{(2)}.\text{date} \le \dots \le t_{(M)}.\text{date}$).

$$S_{\text{cum}}(t_{(k)}) = \sum_{j=1}^k |t_{(j)}.\text{amount}|$$

**Purpose**: Tracks expenditure acceleration over time to evaluate burn rates across billing cycles.

---

## 2. RFM Feature Engineering

The RFM (Recency, Frequency, Monetary) transformation aggregates transactional events into behavioral vectors for each spending category.

### 2.1 Feature Definitions

Let $T_c \subseteq T$ be the subset of expense transactions belonging to category $c$.

1. **Recency ($R_c$)**:
   Let $D_{\text{ref}}$ be the dataset reference date (typically $\max_{t \in T} t.\text{date}$).
   $$R_c = \max\left(0, (D_{\text{ref}} - \max_{t \in T_c} t.\text{date}).\text{days}\right)$$
   *Purpose*: Measures how recently spending occurred in category $c$. Low recency denotes active ongoing spending.

2. **Frequency ($F_c$)**:
   $$F_c = |T_c|$$
   *Purpose*: Measures how often transactions occur in category $c$, distinguishing habitual recurring micro-purchases from one-off payments.

3. **Monetary ($M_c$)**:
   $$M_c = \sum_{t \in T_c} |t.\text{amount}|$$
   *Purpose*: Measures total capital absorbed by category $c$.

### 2.2 Feature Standardization (Z-Score)

Because $M_c$ can be in thousands of dollars while $F_c$ is in single units, unscaled Euclidean distances would be dominated by the monetary dimension. To give each feature equal weight in distance calculations, we standardize each feature column $X \in \{R, F, M\}$ across all $K$ categories.

Sample mean:
$$\mu_X = \frac{1}{K} \sum_{i=1}^K X_i$$

Sample standard deviation (with Bessel's correction $N - 1$):
$$s_X = \sqrt{\frac{1}{K - 1} \sum_{i=1}^K (X_i - \mu_X)^2}$$

Standardized score:
$$Z_{i, X} = \begin{cases} \frac{X_i - \mu_X}{s_X}, & \text{if } s_X > 0 \text{ and } K > 1 \\ 0.0, & \text{otherwise} \end{cases}$$

Each category $c$ is mapped to a normalized 3D coordinate:
$$\mathbf{x}_c = [Z_{c, R},\, Z_{c, F},\, Z_{c, M}]^T \in \mathbb{R}^3$$

**Purpose**: Places all features onto a dimensionless scale with zero mean and unit variance, ensuring isotropic distance calculations.

---

## 3. K-Means Clustering from Scratch

The clustering engine partitions $N$ category vectors into $k$ habit clusters using pure NumPy vector operations without external machine learning libraries.

### 3.1 Distance Metric

For two points $\mathbf{x}, \mathbf{y} \in \mathbb{R}^D$, Euclidean distance and squared Euclidean distance are defined as:

$$\|\mathbf{x} - \mathbf{y}\|_2^2 = \sum_{j=1}^D (x_j - y_j)^2$$

### 3.2 K-Means++ Initialization

To avoid poor local minima from naive random seeding, initial centroids $\mathbf{c}_1, \dots, \mathbf{c}_k$ are selected via distance-weighted sampling:

1. Choose first centroid $\mathbf{c}_1$ uniformly at random from dataset samples:
   $$\mathbf{c}_1 \sim \text{Uniform}(\{\mathbf{x}_1, \dots, \mathbf{x}_N\})$$

2. For each remaining centroid $m \in \{2, \dots, k\}$:
   Compute the shortest squared distance from each point $\mathbf{x}_i$ to any already chosen centroid:
   $$D(\mathbf{x}_i)^2 = \min_{j \in \{1, \dots, m-1\}} \|\mathbf{x}_i - \mathbf{c}_j\|_2^2$$

   Select the next centroid $\mathbf{c}_m$ with probability proportional to $D(\mathbf{x}_i)^2$:
   $$P(\mathbf{x}_i) = \frac{D(\mathbf{x}_i)^2}{\sum_{l=1}^N D(\mathbf{x}_l)^2}$$

**Purpose**: Guarantees wide geometric separation between initial centroids, yielding provably bounded clustering errors ($O(\log k)$ competitive ratio).

### 3.3 Expectation Step (Cluster Assignment)

Each point $\mathbf{x}_i$ is assigned to the nearest cluster centroid:

$$l_i = \arg\min_{j \in \{1, \dots, k\}} \|\mathbf{x}_i - \mathbf{c}_j\|_2^2$$

The cluster partition is:
$$C_j = \{ \mathbf{x}_i : l_i = j \}$$

**Purpose**: Creates Voronoi cells that partition the financial feature space.

### 3.4 Maximization Step (Centroid Update)

Centroids are recomputed as the arithmetic mean of all points assigned to that cluster:

$$\mathbf{c}_j^{(t+1)} = \frac{1}{|C_j|} \sum_{\mathbf{x}_i \in C_j} \mathbf{x}_i$$

If a cluster becomes empty ($|C_j| = 0$), it is re-seeded with a random observation from the dataset to preserve $k$ clusters.

**Purpose**: Minimizes intra-cluster variance for the current assignment partition.

### 3.5 Convergence Criterion

The loop halts when the total displacement of all centroids falls below tolerance $\epsilon = 10^{-4}$ or maximum iterations ($100$) are reached:

$$\Delta C = \sum_{j=1}^k \|\mathbf{c}_j^{(t+1)} - \mathbf{c}_j^{(t)}\|_2 \le \epsilon$$

**Purpose**: Prevents unnecessary iterations once cluster assignments stabilize.

### 3.6 Objective Function (Inertia / WCSS)

The within-cluster sum of squares (WCSS) quantifies cluster compactness:

$$J = \sum_{j=1}^k \sum_{\mathbf{x}_i \in C_j} \|\mathbf{x}_i - \mathbf{c}_j\|_2^2$$

**Purpose**: Evaluates clustering compactness for model comparison and elbow analysis.

---

## 4. Optimal-k Discovery (Secant Chord Elbow Method)

To discover the natural number of spending clusters without manual guessing, the system computes inertia across candidate values $k \in \{1, \dots, \min(k_{\max}, N)\}$.

### 4.1 Unit Coordinate Normalization

Inertia values $J(k)$ and cluster counts $k$ operate on different scales. Both axes are normalized to $[0, 1]$:

$$\hat{k} = \frac{k - k_{\min}}{k_{\max} - k_{\min}}$$

$$\hat{J}(k) = \frac{J(k) - J_{\min}}{J_{\max} - J_{\min}}$$

### 4.2 Distance to Secant Chord

A straight line connects the curve endpoints $(0, 1)$ and $(1, 0)$ in the normalized space. The equation of this chord is:

$$\hat{k} + \hat{J} - 1 = 0$$

The perpendicular distance from any normalized curve point $(\hat{k}, \hat{J}(k))$ to this chord is:

$$d(k) = \frac{|\hat{k} + \hat{J}(k) - 1|}{\sqrt{1^2 + 1^2}} = \frac{|\hat{k} + \hat{J}(k) - 1|}{\sqrt{2}}$$

The recommended optimal cluster count $k^*$ is the inflection point maximizing this distance:

$$k^* = \arg\max_{k} d(k)$$

**Purpose**: Provides an autonomous, scale-invariant selection of $k$ where marginal inertia reduction diminishes most sharply (the elbow).

---

## 5. Habit Persona Classification Logic

Clusters are mapped to client habit personas using robust median thresholds on unscaled RFM metrics:

Let $\tilde{M} = \text{median}(\{M_c\})$ and $\tilde{F} = \text{median}(\{F_c\})$.

For each cluster $j$, let $\bar{M}_j$, $\bar{F}_j$, and $\bar{R}_j$ be the cluster means:

$$\text{Persona}(j) = \begin{cases}
\text{High Volume / Core Lifestyle}, & \text{if } \bar{M}_j \ge \tilde{M} \text{ and } \bar{F}_j \ge \tilde{F} \\
\text{Major Outflow / Fixed Bill}, & \text{if } \bar{M}_j \ge \tilde{M} \text{ and } \bar{F}_j < \tilde{F} \\
\text{Frequent Micro-Spender (Foodie/Daily)}, & \text{if } \bar{M}_j < \tilde{M} \text{ and } \bar{F}_j \ge \tilde{F} \\
\text{Low Spend / Budget Saver}, & \text{if } \bar{M}_j < \tilde{M} \text{ and } \bar{F}_j < \tilde{F}
\end{cases}$$

**Purpose**: Converts numeric coordinates into plain-language financial habit profiles for client discussions.

---

## 6. Prescriptive Counterfactual Intervention and Wealth Compounding

### 6.1 Counterfactual Expenditure Adjustment

When a client simulates a spending cut of $\alpha \in [0, 1]$ (for example, $\alpha = 0.20$ for a 20% reduction) in category $c$:

$$M'_c = M_c \times (1 - \alpha)$$

$$\text{Saved Capital} = M_c - M'_c = M_c \times \alpha$$

Assuming regular monthly budgeting cycles:

$$S_{\text{monthly}} = \text{Saved Capital}$$

$$S_{\text{annual}} = S_{\text{monthly}} \times 12$$

### 6.2 Counterfactual Feature Standardization

To avoid data leakage, the perturbed monetary value is standardized using the **unperturbed baseline statistics** ($\mu_M, s_M$):

$$Z'_{c, M} = \frac{M'_c - \mu_M}{s_M}$$

The counterfactual category vector becomes:
$$\mathbf{x}'_c = [Z_{c, R},\, Z_{c, F},\, Z'_{c, M}]^T$$

The fitted model then predicts the new cluster:
$$j' = \arg\min_{k} \|\mathbf{x}'_c - \mathbf{c}_k\|_2^2$$

**Purpose**: Accurately shows whether an expense reduction causes a habit shift (e.g. migrating from Frequent Micro-Spender to Budget Saver) without altering baseline coordinate references.

### 6.3 Future Value of Annuity (Wealth Compounding)

If the liberated monthly savings $S_{\text{monthly}}$ are redirected into an investment portfolio with expected annual return rate $r_{\text{annual}}$ compounded monthly:

Monthly compounding rate:
$$r_m = \frac{r_{\text{annual}}}{12}$$

Total compounding periods for horizon of $n$ years:
$$N = 12 \times n$$

Future Value ($FV$) of ordinary annuity (contributions at end of month):

$$FV(n) = \begin{cases}
S_{\text{monthly}} \times \left( \frac{(1 + r_m)^N - 1}{r_m} \right), & \text{if } r_m > 0 \\
S_{\text{monthly}} \times N, & \text{if } r_m = 0
\end{cases}$$

The total wealth generated from compound interest alone is:
$$\text{Compounded Interest} = FV(n) - (S_{\text{monthly}} \times N)$$

**Purpose**: Quantifies the long-term opportunity cost of discretionary spending, proving how small lifestyle adjustments generate substantial retirement or investment wealth over 1, 3, 5, 10, and 20 years.

---

## 7. Recurring Transaction Calendar Arithmetic

For recurring commitments with base date $D_0 = (Y_0, M_0, d_0)$ and occurrence index $n \in \mathbb{N}_0$:

1. **Daily**:
   $$D_n = D_0 + n \times 1\text{ day}$$

2. **Weekly**:
   $$D_n = D_0 + n \times 7\text{ days}$$

3. **Monthly**:
   Let total month offset be $m = (M_0 - 1) + n$.
   $$Y_n = Y_0 + \lfloor m / 12 \rfloor$$
   $$M_n = (m \bmod 12) + 1$$
   $$d_n = \min(d_0,\, \text{days\_in\_month}(Y_n, M_n))$$
   $$D_n = (Y_n, M_n, d_n)$$

4. **Yearly**:
   $$Y_n = Y_0 + n$$
   $$d_n = \min(d_0,\, \text{days\_in\_month}(Y_n, M_0))$$
   $$D_n = (Y_n, M_0, d_n)$$

**Purpose**: Accurately accounts for varying month lengths (28, 29, 30, 31 days) and leap years, preventing calendar overflow errors on recurring statements.
