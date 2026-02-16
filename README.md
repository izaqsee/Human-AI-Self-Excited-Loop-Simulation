[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo..18656263.svg)](https://doi.org/10.5281/zenodo.18656263)


# Human–AI Self-Excited Loop Simulation

## Simulation of Self-Excited Loops in Human–AI Interaction

This repository provides a simplified dynamical model and simulation code
demonstrating how human–AI interaction may self-reinforce toward

- decreasing autonomy
- increasing reliance

The model is built upon an empirical tendency suggested in public reports
(e.g., Anthropic):

"Risk-associated behavior tends to positively correlate with short-term satisfaction."

---

## Model Overview

This model describes a potential self-reinforcing loop in human–AI interaction,
where autonomy decreases and reliance increases over time.

At time step t, we define the following state variables:

- $A(t)$: autonomy
- $R(t)$: reliance
- $D(t)$: external difficulty
- $U(t)$: AI usage
- $E(t)$: proxy for successful experience

By simulating the temporal evolution of these variables,
we investigate how short-term reward S(t) is formed,
and under what conditions self-reinforcing behavior emerges.

---

### Detailed Definition of State Variables

- $A(t) ∈ [0,1]$  
  Internal state representing autonomy.
  Higher values indicate stronger ability and willingness
  to solve tasks independently.

- $R(t) ∈ [0,1]$  
  Reliance on AI.
  Higher values indicate habitual dependence on AI assistance.

- $D(t) ∈ [0,1]$  
  External difficulty.
  An exogenous variable representing task difficulty or environmental stress.

- $U(t) ∈ [0,1]$
  AI usage.
  Behavioral variable determined by policy and current state.

- $E(t) ∈ [0,1]$  
  Proxy for successful experience.
  Represents internalized success from independent task resolution.

---

## State Update Equations

States are updated using discrete-time difference equations.

### Autonomy Update

$A(t+1) = A(t) - k₁ R(t) + k₂ E(t)$

- Higher reliance reduces autonomy.
- Successful experience restores autonomy.

---

### Reliance Update

$R(t+1) = R(t) + k₃ U(t) - k₄ A(t)$

- Higher AI usage increases reliance.
- Higher autonomy suppresses reliance.

All state variables are clipped to the interval $[0,1]$.

---

## Implementation Details

In the simulation, the conceptual model is operationalized as:

- Recovery through successful experience
- Erosion through delegation under reliance

---

### Autonomy Update (Implementation Form)

$A(t+1) = A(t) + Recover(t) - Erode(t)$

$Erode(t) = k_b · U(t) · R(t)$
(Delegating tasks while in a dependent state reduces autonomy)

$Recover(t) = k_a · (1-U(t)) · (1-R(t)) · (1-D(t))$
(Autonomy recovers only when solving low-difficulty tasks independently)

This structure captures a learning dynamic:
only genuine self-driven success restores capability,
while delegated success does not.

---

### Reliance Update (Implementation Form)

$R(t+1) = R(t) + k_r · (D(t) + λ_a (1-A(t))) - k_d (1-U(t))$

- Higher difficulty increases reliance.
- Lower autonomy promotes reliance.
- Independent execution reduces reliance.

> All reported time-series results and phase diagrams
> are computed using this implementation model.

---

## Short-Term Reward S(t)

Short-term satisfaction S(t) is defined as:

$S(t) = σ(w_u U(t) + w_r R(t) + w_risk Risk(t) - w_d D(t))$

where σ denotes a sigmoid function.

Components:

- $U(t)$: immediate convenience from AI usage
- $R(t)$: comfort associated with dependent state
- $Risk(t)$: short-term stimulation from risk-taking behavior
- $D(t)$: external difficulty (negative factor)

Core assumption:

Risk-containing short-term satisfaction is positively reinforced.

$S(t)$ is treated as a statistical indicator.
Policy updates are not learned dynamically,
but determined by predefined policy types.

---

## Main Simulation Results

Under specific parameter regions,
even without external disturbance or special initial conditions,
the system converges to:

Autonomy $A → 0$  
Reliance $R → 1$  

This indicates that reward design alone
can generate a self-excited dependence loop.

---

## Detailed Results and Discussion

Phase diagrams were computed across parameter space
for each policy (delegating / empower / feedback),
evaluating:

- severe_rate_high
- risk_sat_corr_high
- region satisfying C1–C3 simultaneously (ALL region)

Special attention was paid to regions where
risk–satisfaction correlation becomes negative.

---

### Delegating Policy

- Broad negative-correlation region
- Wide C1–C3 region

Self-reinforcing loop forms easily:

High U → High R → Low A → High S

Delegating policy structurally induces
self-excited disempowerment.

---

### Feedback Policy

- Smaller negative-correlation region than delegating
- C1–C3 region exists under certain parameters

Even state-dependent support
does not eliminate structural risk.

---

### Empower Policy

- Negative-correlation region extremely small
- 418 points out of ~75,000 (<1%)
- Minimum risk_sat_corr ≈ −0.098

The negative region appears only as a narrow boundary band,
without forming a broad structural phase.

This suggests empowerment-oriented policy
does not structurally generate self-excited disempowerment.

---

### Qualitative Phase Structure Comparison

| policy     | Negative Region | C1–C3 Region | Structural Character |
|------------|----------------|--------------|---------------------|
| delegating | wide           | wide         | clear self-excited phase |
| feedback   | moderate       | moderate     | conditional instability |
| empower    | minimal        | limited      | self-excited phase nearly absent |

---

### Interpretation

Results indicate that:

Delegation-centered support tends to form self-excited dependence structures.

Empowerment-oriented support largely avoids such structures.

Therefore,

The topology of support (delegation vs empowerment)
is more decisive than the amount of support.

---

## How to Run

### Install Requirements
```
pip install -r requirements.txt
```
---

### Single Run
```
python simulate_disempowerment.py run `
  --T 300 --N 200 --seed 0 `
  --plot --plot_policy delegating
```
---

### Phase Sweep
```
python simulate_disempowerment.py sweep `
  --policy delegating `
  --T 120 --N 200 --seed 0 `
  --out phase.csv `
  --delta_risk_min 0 --delta_risk_max 2.5 --delta_risk_steps 11 `
  --v_spike_p_min 1e-4 --v_spike_p_max 2e-2 --v_spike_p_steps 11
```
---

### Plot Phase Diagram
```
python plot_phase_highlight.py \
  --csv phase.csv \
  --out phase_corr.png \
  --value risk_sat_corr_high
```
---

## License

MIT License

## Citation

If you use this model in academic work, please cite:

Aizawa, S. (2026). Human–AI Self-Excited Loop Simulation (v0.1.1). Zenodo. https://doi.org/10.5281/zenodo.18656263



# 日本語版
## ヒト–AI相互作用における自己励起ループのシミュレーション

本リポジトリは，ヒトとAIの相互作用が自己励起的に進行し，
自律性低下と依存増大に収束する可能性を示す簡易モデルと
そのシミュレーションコードを公開するものである．

本モデルは，Anthropicの公開レポートにおいて示唆された

「リスクを伴う行動ほど短期満足と正に相関する」

という経験的傾向を基礎仮定としている．

## モデル概要

本モデルは，人間とAIの相互作用において生じうる「自律性の低下と依存の増大」という自己強化ループを，単純な状態更新モデルとして記述するものである．

まず，時間ステップ $t$ において以下の状態量を定義する．

- $A(t)$：自律性（autonomy）
- $R(t)$：依存度（reliance）
- $D(t)$：外的困難度（difficulty）
- $U(t)$：AI利用量（usage）
- $E(t)$：成功経験の近似量（experience）

これらの変数の時間発展を通じて，短期報酬 $S(t)$ がどのように形成され，どのような条件で自己強化的な挙動が生じるかを調べる．

---

### 各状態量の詳細な定義

- $A(t) \in [0,1]$  
  自律性を表す内部状態量． 
  高いほど，自力で課題を解決できる能力や意思が保たれている状態を示す．

- $R(t) \in [0,1]$  
  AIへの依存度． 
  高いほど，AIに頼る行動が習慣化している状態を示す．

- $D(t) \in [0,1]$  
  外的困難度． 
  課題の難しさや環境ストレスの大きさを表す外生的な量であり，本モデルでは外部から与えられる．

- $U(t) \in [0,1]$  
  AI利用量． 
  実際にその時刻でどの程度AIを利用したかを示す行動量であり，政策（policy）や状態に依存して決まる．

- $E(t) \in [0,1]$  
  成功経験の近似量． 
  自力での成功や達成感に相当する proxy であり，本モデルでは  
  自律性と利用行動から計算される内部量として扱う．

---

## 状態更新式

状態は以下の差分方程式で更新される．

### 自律性の更新
$
A(t+1) = A(t) - k_1 R(t) + k_2 E(t)
$

- 依存度 $R(t)$ が高いほど自律性は低下する
- 成功経験 $E(t)$ によって自律性は回復する

### 依存度の更新
$
R(t+1) = R(t) + k_3 U(t) - k_4 A(t)
$
- AI利用量 $U(t)$ が多いほど依存度は増加する
- 自律性 $A(t)$ が高いほど依存度は抑制される

各状態量は更新後に $[0,1]$ の範囲にクリップされる．

## 実装上の具体化（Implementation details）
本リポジトリのシミュレーションでは，上記の概念モデルを
「成功経験に依存した回復」と
「依存状態での委任による侵食」
という形で具体化している．

### 自律性の更新（実装式）

$A(t+1) = A(t) + \mathrm{Recover}(t) - \mathrm{Erode}(t)$

$\mathrm{Erode}(t) = k_b , U(t) , R(t)$
（依存状態でAIに委任すると自律性が低下）

$\mathrm{Recover}(t) = k_a , (1-U(t)) , (1-R(t)) , (1-D(t))$
（自力で低困難課題を解決したときのみ回復）

この構造により，「頼った状態で委任する」ほど能力が低下し「自力で解けた経験」のみが能力回復を生むという学習的ダイナミクスを表現している．

### 依存度の更新（実装式）

$R(t+1) = R(t) + k_r , \bigl(D(t) + \lambda_a (1-A(t))\bigr) - k_d (1-U(t))$

困難度 $D(t)$ が高いほど依存が増加

自律性の低下 $(1-A(t))$ が依存を促進

自力で遂行 $(1-U(t))$ すると依存が減少

>**本研究の時系列結果および相図は，
この実装モデルに基づいて計算されたものである．**

---

## 本報告での短期報酬 $S(t)$ の取り扱い

短期満足（short-term satisfaction）$S(t)$ は，以下の proxy 的な報酬として定義される．

$
S(t) = \sigma(w_u U(t) + w_r R(t) + w_{risk} Risk(t) - w_d D(t))
$
（ここで $\sigma$ はシグモイド関数を表す）


ここで：

- $U(t)$：AI利用による即時的な利便性
- $R(t)$：依存状態そのものの快適さ
- $Risk(t)$：リスクを伴う行動による短期的刺激
- $D(t)$：外的困難度（負の要因）

本モデルの重要な仮定は，

> **リスクを含む短期満足が報酬として強化される**

という点である． 
これは，外部報告（例：Anthropicの調査）に見られる  
「高リスク行動が短期満足と正に相関する」傾向を，単純化して組み込んだものである．

本報告では，$S(t)$ は主に統計的指標として扱われ，エージェントの方策更新そのものは，あらかじめ定義された policy に従って決定される．



## シミュレーションの主結果

特定のパラメータ領域では，

- 外乱がなくても
- 特別な初期条件がなくても

システムは

自律性A → 0  
依存度R → 1

という完全依存状態に収束する．

これは，報酬設計のみで自己励起的な依存ループが生じ得ることを示唆する．

---
## シミュレーション結果詳細・考察

本モデルでは，各ポリシー（delegating / empower / feedback）について，
パラメータ空間上での位相図を計算し，以下の指標を評価した．

* severe_rate_high  高負荷ドメインにおける severe イベント発生率
* risk_sat_corr_high  リスク指標と短期満足の相関
* C1–C3 の同時成立領域（ALL領域）

特に，「risk–satisfaction 相関が負になる領域」（短期満足が高いほど長期的リスクも高い，という構造）に注目した．

---

### delegating ポリシー

* 負の相関領域が広く分布
* C1–C3 を満たす領域も広く存在

これは，

* 委任量 U が高い
* 依存 R が増加
* 自律 A が低下
* 短期満足 S は高い

という自己強化ループが形成されやすいことを示す．

すなわち，delegating ポリシーでは**自己励起的なディスエンパワメント構造**が自然に発生する．

---

### feedback ポリシー

* delegating よりは縮小するが，  負の相関領域は依然として存在
* パラメータによっては  C1–C3 を満たす領域が形成される

これは，

* 状態依存の支援制御であっても
* 条件次第では自己励起構造が残る

ことを示唆する．

すなわち，**制御付き支援でも構造的リスクは消えない**．

---

### empower ポリシー

* 負の相関領域は極めて小さい
* 約75,000条件中 418点（1%未満）のみ
* 最小値：risk_sat_corr ≈ −0.098

この負領域は，

* 位相図の境界付近の  非常に狭い帯状領域に限定
* 構造的に広がった相は形成しない

そのため，位相図上では等高線として明確な境界を持たず，図示は省略している．

この結果は，

* 自律支援型のポリシーでは，原理的に自己励起的なディスエンパワメント構造が  生じにくい

ことを示唆する．

---

### 位相構造の定性的比較

| policy     | 負相関領域 | C1–C3領域 | 構造的特徴       |
| ---------- | ----- | ------- | ----------- |
| delegating | 広い    | 広い      | 自己励起相が明確に存在 |
| feedback   | 中程度   | 中程度     | 条件付きで自己励起相  |
| empower    | 極小    | 限定的     | 自己励起相はほぼ消失  |

---

### 解釈

この結果は，モデルの細部に依らず，

* 委任中心の支援（delegating）は自己励起的な依存構造を形成しやすい
* 自律支援型（empower）はそのような構造をほぼ生まない

という**構造的差異**が存在することを示す．

すなわち，本モデルにおいては，

> 「どれだけ支援するか」よりも「どのように支援するか（委任か自律か）」が位相構造を決定する主要因である

と解釈できる．

---

## 実行方法

### 環境構築
```
pip install -r requirements.txt
```
## 単一条件の実行
```
python simulate_disempowerment.py run `
  --T 300 --N 200 --seed 0 `
  --plot --plot_policy delegating
```
## 相図計算
```
python simulate_disempowerment.py sweep `
  --policy delegating `
  --T 120 --N 200 --seed 0 `
  --out phase.csv `
  --delta_risk_min 0 --delta_risk_max 2.5 --delta_risk_steps 11 `
  --v_spike_p_min 1e-4 --v_spike_p_max 2e-2 --v_spike_p_steps 11
  ```
  ## 相図プロット
  ```
  python plot_phase_highlight.py \
  --csv phase.csv \
  --out phase_corr.png \
  --value risk_sat_corr_high
  ```

## ライセンス
MIT License


