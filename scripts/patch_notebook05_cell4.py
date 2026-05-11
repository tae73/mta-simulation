"""Fix notebook 05 cell [4] — uses 'Causal' string but new category is 'Causal (incremental)'."""
import nbformat

NB = "notebooks/part1/05_experiments_and_insights.ipynb"
nb = nbformat.read(NB, as_version=4)

new_src = """# 카테고리별 평균 성능
exp01["category"] = exp01["method"].map(METHOD_CATEGORIES)
cat_summary = (
    exp01.groupby("category")
    .agg(mean_mae=("mae", "mean"), best_mae=("mae", "min"),
         mean_tau=("kendall_tau", "mean"), n_methods=("method", "count"))
    .sort_values("mean_mae")
)
print("카테고리별 평균 성능:")
print(cat_summary.to_string(float_format="%.4f"))

# Option 1: split causal into incremental vs debiased
inc_label = "Causal (incremental)"
deb_label = "Causal (debiased)"
rb_mae = cat_summary.loc["Rule-based", "mean_mae"]
if inc_label in cat_summary.index:
    inc_mae = cat_summary.loc[inc_label, "mean_mae"]
    print(f"\\n→ {inc_label} 평균 MAE({inc_mae:.4f}) vs Rule-based({rb_mae:.4f}): "
          f"{(1 - inc_mae / rb_mae):+.0%} 차이 (model-based incremental)")
if deb_label in cat_summary.index:
    deb_mae = cat_summary.loc[deb_label, "mean_mae"]
    print(f"→ {deb_label} 평균 MAE({deb_mae:.4f}) vs Rule-based({rb_mae:.4f}): "
          f"{(1 - deb_mae / rb_mae):+.0%} 차이 (truly causal: IPW/DR/DML)")"""

nb.cells[4].source = new_src
nbformat.write(nb, NB)
print("Updated cell [4]")
