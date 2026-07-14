# local_name Label Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the API `local_name` (native-language security name) on `Asset` and let `AssetList` / `Portfolio` / `EfficientFrontier` use it as a chart/table label, behind one consistent label selector, without breaking any existing parameter.

**Architecture:** `Asset` reads `local_name` (nullable) and stores the raw payload as `Asset.info`. The base `ListMaker` builds a parallel `local_names` map (with `or name` fallback) in the same loop as `names`, and exposes one resolver `_asset_labels(mode)`. Every plotting/reporting site routes through the resolver; legacy params (`tickers`, `ticker_names`, `labels_are_tickers`) are silently mapped to the new label-mode vocabulary.

**Tech Stack:** Python 3.14 env (library min per `pyproject.toml`), pandas, numpy, matplotlib; pytest + pytest-mock; poetry; ruff.

## Global Constraints

- All comments/docstrings in **English**.
- Modern syntax: built-in generics (`list[str]`, `dict[str, str]`), `X | None`; literals over constructors.
- TDD: failing test first, then minimal code; `pytest -q` clean after each task.
- `poetry run ruff check .` clean before finishing (targeted `# noqa: <CODE>` only if unavoidable, with rationale).
- Library code must run on the `pyproject.toml` Python minimum; notebooks must run on Colab 3.12.
- Do **not** commit `poetry.lock`.
- Label-mode vocabulary is exactly `"ticker" | "name" | "local_name"` (singular). Public `plot_assets`/`plot_pair_ef` string values remain plural: `"tickers" | "names" | "local_names"`.
- **Alignment invariant:** `_asset_labels(mode)[i]` must correspond to `self.symbols[i]` for every mode (frontier does `dict(zip(labels, weights))`).
- Run commands from the worktree: `/home/chilango/pyprojects/okama_projects/okama/.claude/worktrees/issue-93-local-name`. Prefix with `poetry run` if not in the poetry shell.

---

### Task 1: `Asset.local_name` + `Asset.info` + repr

**Files:**
- Modify: `okama/asset.py` (`_get_symbol_data` ~83-92; `__repr__` ~62-75)
- Test: `tests/test_asset.py`

**Interfaces:**
- Produces: `Asset.info: dict` (raw payload), `Asset.name: str`, `Asset.local_name: str | None`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_asset.py`. Extend `_DefaultMocks.symbol_info` to optionally carry `local_name` and add tests:

```python
def test_local_name_present(basic_patches):
    dm = basic_patches["defaults"]
    dm.symbol_info["local_name"] = "Сбербанк"
    basic_patches["m_get_symbol_info"].return_value = dm.symbol_info
    a = ok.Asset("SPY.US")
    assert a.local_name == "Сбербанк"
    assert a.info == dm.symbol_info          # raw payload stored
    assert "local_name" in repr(a)


def test_local_name_absent_is_none(basic_patches):
    # default symbol_info has no "local_name" key
    a = ok.Asset("SPY.US")
    assert a.local_name is None
    assert a.info["name"] == "SPDR S&P 500 ETF Trust"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `poetry run pytest tests/test_asset.py::test_local_name_present tests/test_asset.py::test_local_name_absent_is_none -q`
Expected: FAIL (`AttributeError: 'Asset' object has no attribute 'local_name'`).

- [ ] **Step 3: Implement**

In `okama/asset.py` `_get_symbol_data`:

```python
def _get_symbol_data(self, symbol) -> None:
    x = data_queries.QueryData.get_symbol_info(symbol)
    self.info: dict = x
    self.ticker: str = x["code"]
    self.name: str = x["name"]
    self.local_name: str | None = x.get("local_name")
    self.country: str = x["country"]
    self.exchange: str = x["exchange"]
    self.currency: str = x["currency"]
    self.type: str = x["type"]
    self.isin: str = x["isin"]
    self.inflation: str = f"{self.currency}.INFL"
```

In `__repr__`, add `"local_name": self.local_name,` right after the `"name"` entry.

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_asset.py -q`
Expected: PASS (all asset tests, including pre-existing).

- [ ] **Step 5: Commit**

```bash
git add okama/asset.py tests/test_asset.py
git commit -m "feat(asset): expose local_name and raw info payload on Asset"
```

---

### Task 2: `FakeAsset.local_name` test fixture attribute

**Files:**
- Modify: `tests/helpers/factories.py` (`FakeAsset.__init__` ~36-54)

**Interfaces:**
- Produces: `FakeAsset(symbol, ror, currency=..., name=..., local_name=None)` with `self.local_name`.

This is a test-infra change (no production logic) enabling Tasks 3-6.

- [ ] **Step 1: Modify `FakeAsset`**

Change the signature and body:

```python
def __init__(
    self,
    symbol: str,
    ror: pd.Series,
    currency: str = "USD",
    name: str | None = None,
    local_name: str | None = None,
):
    self.symbol = symbol
    self.ticker = symbol.split(".")[0]
    self.name = name or f"{self.ticker} name"
    self.local_name = local_name
    self.currency = currency
    ...
```

(Leave the rest of `__init__` unchanged. `FakeCurrencyAsset` is intentionally **not** given `local_name` — the currency asset is never in `asset_obj_dict`, so `_make_list` never reads it.)

- [ ] **Step 2: Run existing collection tests to verify nothing broke**

Run: `poetry run pytest tests/test_list_maker.py -q`
Expected: PASS (unchanged behavior; new optional param defaults to `None`).

- [ ] **Step 3: Commit**

```bash
git add tests/helpers/factories.py
git commit -m "test: add optional local_name to FakeAsset factory"
```

---

### Task 3: `ListMaker.local_names` map + `_asset_labels` resolver + alignment invariant

**Files:**
- Modify: `okama/common/make_asset_list.py` (`_make_list` ~218-289; `__init__` unpack ~56-66; add resolver + helper near other methods)
- Test: `tests/test_list_maker.py`

**Interfaces:**
- Consumes: `Asset.local_name` (Task 1), `FakeAsset.local_name` (Task 2).
- Produces:
  - `self.local_names: dict[str, str]` on `AssetList` / `Portfolio` / `EfficientFrontier`.
  - `self._asset_labels(mode: str = "ticker") -> list[str]`.
  - `staticmethod ListMaker._labels_mode_from_bool(ticker_names: bool) -> str`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_list_maker.py` (uses the `synthetic_env` fixture which mocks `_get_asset_obj_dict` with `FakeAsset`s). First give two of the fake assets a `local_name`, then assert:

```python
def test_local_names_fallback_mixed_basket(mocker):
    # IDX.US has a native name; A.US does not -> falls back to its Latin name
    from tests.helpers.factories import FakeAsset
    idx = pd.period_range("2020-01", periods=24, freq="M")
    import numpy as np
    rng = np.random.default_rng(1)
    a1 = pd.Series(rng.normal(0.01, 0.05, 24), index=idx, name="IDX.US")
    a2 = pd.Series(rng.normal(0.01, 0.04, 24), index=idx, name="A.US")
    fakes = {
        "IDX.US": FakeAsset("IDX.US", a1, name="Index Fund", local_name="Индекс"),
        "A.US": FakeAsset("A.US", a2, name="Asset A", local_name=None),
    }
    mocker.patch(
        "okama.common.make_asset_list.ListMaker._get_asset_obj_dict",
        side_effect=lambda symbols, first_date=None, last_date=None: {s: fakes[s] for s in symbols},
    )
    from tests.helpers.factories import FakeCurrencyAsset
    mocker.patch("okama.common.make_asset_list.asset.Asset", side_effect=FakeCurrencyAsset)
    al = ok.AssetList(["IDX.US", "A.US"], inflation=False)
    assert al.local_names == {"IDX.US": "Индекс", "A.US": "Asset A"}


def test_asset_labels_modes_and_alignment(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US", "B.US"], inflation=False)
    assert al._asset_labels("ticker") == al.symbols
    assert al._asset_labels("name") == list(al.names.values())
    assert al._asset_labels("local_name") == list(al.local_names.values())
    # positional alignment: label[i] belongs to symbol[i]
    for i, sym in enumerate(al.symbols):
        assert al._asset_labels("name")[i] == al.names[sym]
        assert al._asset_labels("local_name")[i] == al.local_names[sym]
    with pytest.raises(ValueError):
        al._asset_labels("bad")
```

(Ensure `import pytest` is present in the test module.)

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_list_maker.py::test_local_names_fallback_mixed_basket tests/test_list_maker.py::test_asset_labels_modes_and_alignment -q`
Expected: FAIL (`AttributeError: ... 'local_names'` / `_asset_labels`).

- [ ] **Step 3: Implement in `_make_list`**

Add a `local_names` dict next to `names` (~line 220):

```python
names: dict[str, str] = {}
local_names: dict[str, str] = {}
```

Inside the `for asset_item in self.asset_obj_dict.values():` loop, next to `names[...]` (~line 249):

```python
names[asset_item.symbol] = asset_item.name
local_names[asset_item.symbol] = asset_item.local_name or asset_item.name
```

Add `"local_names_dict": local_names,` to the returned dict (after `"names_dict"`).

- [ ] **Step 4: Unpack in `__init__`**

Update the tuple unpacking (~56-66) to include `self.local_names` right after `self.names`. The returned dict order must match; insert `"local_names_dict"` in `_make_list`'s return dict immediately after `"names_dict"`, and add `self.local_names,` immediately after `self.names,` in the unpack tuple.

- [ ] **Step 5: Add resolver + helper**

Add methods to `ListMaker` (place near `__len__`/`__getitem__`, e.g. after `__getitem__`):

```python
def _asset_labels(self, mode: str = "ticker") -> list[str]:
    """Return asset labels aligned with self.symbols for the given label mode."""
    if mode == "ticker":
        return list(self.symbols)
    if mode == "name":
        return list(self.names.values())
    if mode == "local_name":
        return list(self.local_names.values())
    raise ValueError(f"Unknown label mode: {mode!r}. Use 'ticker', 'name' or 'local_name'.")

@staticmethod
def _labels_mode_from_bool(ticker_names: bool) -> str:
    """Map the legacy ticker_names bool to a label mode ('ticker'/'name')."""
    return "ticker" if ticker_names else "name"
```

- [ ] **Step 6: Run tests to verify pass**

Run: `poetry run pytest tests/test_list_maker.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add okama/common/make_asset_list.py tests/test_list_maker.py
git commit -m "feat(list): add local_names map and _asset_labels resolver"
```

---

### Task 4: `plot_assets` (+ `plot_pair_ef`) accept `local_names`

**Files:**
- Modify: `okama/common/make_asset_list.py` (`plot_assets` label block ~734-744; docstring ~674-679)
- Modify: `okama/frontier/single_period.py` (`plot_pair_ef` docstring ~1131-1135)
- Modify: `okama/frontier/multi_period.py` (`plot_pair_ef` docstring ~1429+)
- Test: `tests/asset_list/test_asset_list.py`

**Interfaces:**
- Consumes: `self._asset_labels` (Task 3).
- Produces: `plot_assets(tickers="tickers"|"names"|"local_names"|list)`.

- [ ] **Step 1: Write failing test**

Add to `tests/asset_list/test_asset_list.py` (use `synthetic_env`). Set local names on the fakes via a local patch, or assert against `local_names` map. Minimal behavioral test on the returned Axes annotations:

```python
def test_plot_assets_local_names_labels(synthetic_env, mocker):
    import matplotlib
    matplotlib.use("Agg")
    al = ok.AssetList(["IDX.US", "A.US"], inflation=False)
    # give deterministic local names via the resolver's source
    al.local_names = {"IDX.US": "Индекс", "A.US": "Актив"}
    ax = al.plot_assets(tickers="local_names")
    texts = [t.get_text() for t in ax.texts]
    assert "Индекс" in texts and "Актив" in texts
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/asset_list/test_asset_list.py::test_plot_assets_local_names_labels -q`
Expected: FAIL (`"local_names"` falls into the list branch → `ValueError: tickers parameter should be a list...`).

- [ ] **Step 3: Implement label block**

Replace the label-selection block in `plot_assets` (~734-744):

```python
# Set the labels
if tickers == "tickers":
    asset_labels = self._asset_labels("ticker")
elif tickers == "names":
    asset_labels = self._asset_labels("name")
elif tickers == "local_names":
    asset_labels = self._asset_labels("local_name")
else:
    if not isinstance(tickers, list):
        raise ValueError("tickers parameter should be a list of string labels.")
    if len(tickers) != len(self.symbols):
        raise ValueError("labels and tickers must be of the same length")
    asset_labels = tickers
```

Update the `tickers` docstring in `plot_assets` to list `'local_names'` (native-language names, e.g. 'Сбербанк' / '贵州茅台'). Update `plot_pair_ef` docstrings (single + multi) to mention `'local_names'`.

- [ ] **Step 4: Run to verify pass**

Run: `poetry run pytest tests/asset_list/test_asset_list.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add okama/common/make_asset_list.py okama/frontier/single_period.py okama/frontier/multi_period.py tests/asset_list/test_asset_list.py
git commit -m "feat(plot_assets): accept 'local_names' label option"
```

---

### Task 5: `AssetList.describe(tickers=)` accepts label mode

**Files:**
- Modify: `okama/asset_list.py` (`describe` signature ~794; rename block ~934-937; docstring ~821-822)
- Test: `tests/asset_list/test_asset_list.py`

**Interfaces:**
- Consumes: `self._asset_labels` (Task 3).
- Produces: `describe(tickers: bool | str = True)`.

- [ ] **Step 1: Write failing test**

```python
def test_describe_local_names_header(synthetic_env):
    al = ok.AssetList(["IDX.US", "A.US"], inflation=False)
    al.local_names = {"IDX.US": "Индекс", "A.US": "Актив"}
    df = al.describe(tickers="local_names")
    assert "Индекс" in df.columns and "Актив" in df.columns
    # legacy bool still works
    df_names = al.describe(tickers=False)
    assert al.names["IDX.US"] in df_names.columns
    df_tickers = al.describe(tickers=True)
    assert "IDX.US" in df_tickers.columns
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/asset_list/test_asset_list.py::test_describe_local_names_header -q`
Expected: FAIL (`tickers="local_names"` is truthy → treated as `True`, no rename, native names absent).

- [ ] **Step 3: Implement**

Change the trailing rename block (~934-937). Resolve a mode from `tickers`:

```python
# resolve label mode from the tickers parameter (bool legacy or mode string)
if isinstance(tickers, bool):
    mode = "ticker" if tickers else "name"
elif tickers == "tickers":
    mode = "ticker"
elif tickers == "names":
    mode = "name"
elif tickers == "local_names":
    mode = "local_name"
else:
    raise ValueError("tickers must be a bool or one of 'tickers', 'names', 'local_names'.")
if mode != "ticker":
    labels = dict(zip(self.symbols, self._asset_labels(mode)))  # noqa: B905
    for ti in self.symbols:
        description = description.rename(columns={ti: labels[ti]})
return description
```

Update the `tickers` docstring to note it accepts a bool or `'tickers'`/`'names'`/`'local_names'`. Keep the `# noqa: C901` on the `def` line.

- [ ] **Step 4: Run to verify pass**

Run: `poetry run pytest tests/asset_list/test_asset_list.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add okama/asset_list.py tests/asset_list/test_asset_list.py
git commit -m "feat(describe): accept local_names label mode, keep bool compat"
```

---

### Task 6: `EfficientFrontier` label mode (single + multi) with legacy shims

**Files:**
- Modify: `okama/frontier/single_period.py` (ctor ~78-94; `get_assets_tickers` ~706-711; add `labels`/`labels_are_tickers` properties)
- Modify: `okama/frontier/multi_period.py` (ctor ~95-111; `ticker_names` property/setter ~275-292; inline label sites ~399,414,805,884,929,1353,1411; add `labels` property)
- Test: `tests/test_frontier.py`

**Interfaces:**
- Consumes: `self._asset_labels`, `self._labels_mode_from_bool` (Task 3).
- Produces on both frontier classes:
  - `self._labels_mode: str`
  - property `labels` (get/set) over `"ticker"|"name"|"local_name"`
  - compat: `ticker_names` (multi, bool property+setter), `labels_are_tickers` (single, bool property)

- [ ] **Step 1: Write failing tests**

Add to `tests/test_frontier.py` (use `synthetic_env`; both classes constructed with the 3 fakes). Cover mode selection, legacy round-trip, and weight alignment:

```python
def test_frontier_labels_mode_single(synthetic_env):
    ef = ok.EfficientFrontierSingle(["IDX.US", "A.US", "B.US"], inflation=False, n_points=3)
    ef.local_names = {"IDX.US": "Индекс", "A.US": "Актив", "B.US": "Б"}
    ef.labels = "local_name"
    assert ef.get_assets_tickers() == ["Индекс", "Актив", "Б"]
    # legacy shim round-trip
    ef.labels_are_tickers = True
    assert ef._labels_mode == "ticker"
    assert ef.labels_are_tickers is True
    ef.labels_are_tickers = False
    assert ef._labels_mode == "name"


def test_frontier_labels_mode_multi(synthetic_env):
    ef = ok.EfficientFrontier(["IDX.US", "A.US", "B.US"], inflation=False)
    ef.local_names = {"IDX.US": "Индекс", "A.US": "Актив", "B.US": "Б"}
    ef.labels = "local_name"
    assert ef._asset_labels(ef._labels_mode) == ["Индекс", "Актив", "Б"]
    # legacy ticker_names round-trip + validation
    ef.ticker_names = False
    assert ef._labels_mode == "name" and ef.ticker_names is False
    ef.ticker_names = True
    assert ef._labels_mode == "ticker" and ef.ticker_names is True
    with pytest.raises(ValueError):
        ef.ticker_names = "yes"


def test_frontier_ctor_ticker_names_default(synthetic_env):
    ef = ok.EfficientFrontierSingle(["IDX.US", "A.US", "B.US"], inflation=False, n_points=3)
    assert ef._labels_mode == "ticker"        # default ticker_names=True
    ef2 = ok.EfficientFrontierSingle(["IDX.US", "A.US", "B.US"], inflation=False, n_points=3, ticker_names=False)
    assert ef2._labels_mode == "name"
```

(Class names confirmed: `EfficientFrontierSingle` = single-period (`single_period.py`), `EfficientFrontier` = multi-period rebalanced (`multi_period.py`).)

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/test_frontier.py -k labels -q`
Expected: FAIL (`labels` / `_labels_mode` missing).

- [ ] **Step 3: Implement — single_period**

In `__init__`, replace `self.labels_are_tickers = ticker_names` (~94) with:

```python
self._labels_mode: str = ListMaker._labels_mode_from_bool(ticker_names)
```

(Ensure `ListMaker` is importable in that module; it inherits from it, so use `self._labels_mode_from_bool(ticker_names)`.)

Add properties (place near other properties):

```python
@property
def labels(self) -> str:
    """Label mode for reports/charts: 'ticker', 'name' or 'local_name'."""
    return self._labels_mode

@labels.setter
def labels(self, mode: str) -> None:
    if mode not in ("ticker", "name", "local_name"):
        raise ValueError("labels must be 'ticker', 'name' or 'local_name'.")
    self._labels_mode = mode

@property
def labels_are_tickers(self) -> bool:
    """Legacy flag: True shows tickers, False shows names."""
    return self._labels_mode == "ticker"

@labels_are_tickers.setter
def labels_are_tickers(self, value: bool) -> None:
    self._labels_mode = self._labels_mode_from_bool(value)
```

Replace `get_assets_tickers` (~706-711) body with:

```python
def get_assets_tickers(self) -> list:
    return self._asset_labels(self._labels_mode)
```

- [ ] **Step 4: Implement — multi_period**

In `__init__`, `self.ticker_names = ticker_names` (~111) currently goes through the setter. Replace the storage: introduce `self._labels_mode` and rewrite the property/setter (~275-292):

```python
@property
def ticker_names(self) -> bool:
    """Legacy flag: True shows tickers, False shows full names."""
    return self._labels_mode == "ticker"

@ticker_names.setter
def ticker_names(self, tickers: bool) -> None:
    if not isinstance(tickers, bool):
        raise ValueError("tickers should be True or False")
    self._labels_mode = self._labels_mode_from_bool(tickers)

@property
def labels(self) -> str:
    """Label mode for reports/charts: 'ticker', 'name' or 'local_name'."""
    return self._labels_mode

@labels.setter
def labels(self, mode: str) -> None:
    if mode not in ("ticker", "name", "local_name"):
        raise ValueError("labels must be 'ticker', 'name' or 'local_name'.")
    self._labels_mode = mode
```

Because `__init__` does `self.ticker_names = ticker_names` (setter) before `_labels_mode` exists, the setter now assigns `self._labels_mode` directly — safe (the setter creates it). Verify `__init__` calls the setter after `super().__init__()`.

Replace every inline site (~399, 414, 805, 884, 929, 1353, 1411):

```python
asset_labels = self.symbols if self.ticker_names else list(self.names.values())
```
with:
```python
asset_labels = self._asset_labels(self._labels_mode)
```

- [ ] **Step 5: Run to verify pass**

Run: `poetry run pytest tests/test_frontier.py tests/test_frontier_reb.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add okama/frontier/single_period.py okama/frontier/multi_period.py tests/test_frontier.py
git commit -m "feat(frontier): add labels mode selector, keep ticker_names compat"
```

---

### Task 7: `Portfolio.table` conditional `local name` column

**Files:**
- Modify: `okama/portfolios/core.py` (`table` ~1491-1498; docstring ~1483-1489)
- Test: `tests/portfolio/test_portfolio.py`

**Interfaces:**
- Consumes: `self.local_names`, `self.asset_obj_dict` (base class).
- Produces: `Portfolio.table` with an extra `local name` column iff any asset has a non-null `local_name`.

- [ ] **Step 1: Write failing tests**

```python
def test_table_adds_local_name_when_present(synthetic_env):
    pf = ok.Portfolio(["IDX.US", "A.US"], weights=[0.5, 0.5], inflation=False)
    # simulate one asset carrying a native name
    pf.asset_obj_dict["IDX.US"].local_name = "Индекс"
    pf.local_names = {"IDX.US": "Индекс", "A.US": "Asset A name"}
    table = pf.table
    assert "local name" in table.columns
    assert list(table["local name"]) == ["Индекс", "Asset A name"]


def test_table_no_local_name_column_when_absent(synthetic_env):
    pf = ok.Portfolio(["IDX.US", "A.US"], weights=[0.5, 0.5], inflation=False)
    # fakes have local_name=None by default
    assert "local name" not in pf.table.columns
    assert list(pf.table.columns) == ["asset name", "ticker", "weights"]
```

- [ ] **Step 2: Run to verify failure**

Run: `poetry run pytest tests/portfolio/test_portfolio.py::test_table_adds_local_name_when_present -q`
Expected: FAIL (no `local name` column).

- [ ] **Step 3: Implement**

Replace the `table` body build (~1491-1498):

```python
x = pd.DataFrame(
    data={
        "asset name": list(self.names.values()),
        "ticker": list(self.names.keys()),
    }
)
if any(a.local_name for a in self.asset_obj_dict.values()):
    x.insert(1, "local name", list(self.local_names.values()))
x["weights"] = self.weights
return x
```

Add a second docstring example showing the 4-column form for a MOEX ticker (e.g. `SBER.MOEX` → `local name` `Сбербанк`).

- [ ] **Step 4: Run to verify pass**

Run: `poetry run pytest tests/portfolio/test_portfolio.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add okama/portfolios/core.py tests/portfolio/test_portfolio.py
git commit -m "feat(portfolio): add local name column to table when available"
```

---

### Task 8: Full suite + ruff + docs

**Files:**
- Modify: docs (Sphinx autodoc picks up `Asset.local_name`/`Asset.info` automatically; add a short note where label options are documented — e.g. `docs/` asset/frontier pages if hand-written)
- Verify: whole repo

- [ ] **Step 1: Run full test suite**

Run: `poetry run pytest -q`
Expected: PASS, no warnings/errors.

- [ ] **Step 2: Ruff**

Run: `poetry run ruff check .`
Expected: clean. Fix any issue; only a targeted `# noqa: <CODE>` with rationale if genuinely unavoidable.

- [ ] **Step 3: Notebook back-compat check (affected notebooks only)**

The notebooks use `ticker_names=`, `describe(tickers=`, `plot_assets(tickers=`, `plot_pair_ef(tickers=`, `.table`. Run nbmake on the affected ones (network required — run only if the environment allows API access; otherwise note as a manual check):

Run: `poetry run pytest --nbmake "examples/06 efficient frontier single period.ipynb" "examples/07 efficient frontier multi-period.ipynb" "examples/03 investment portfolios.ipynb" -q`
Expected: PASS (no signature breaks). If offline, document that this must be run before release.

- [ ] **Step 4: Docs note for CJK fonts**

Where the label option is documented, add: native (`local_name`) labels in CJK scripts require a CJK-capable matplotlib font; default fonts render tofu boxes.

- [ ] **Step 5: Commit**

```bash
git add docs
git commit -m "docs: document local_name, label option and CJK font note"
```

---

## Self-Review

- **Spec coverage:** Asset.local_name/info/repr → Task 1; local_names map + resolver → Task 3; plot_assets/plot_pair_ef → Task 4; describe → Task 5; frontier (both, with both legacy shims) → Task 6; Portfolio.table (conditional) → Task 7; wealth_indexes/drawdowns explicitly excluded (spec); tests incl. alignment invariant → Tasks 1,3,6; FakeAsset infra → Task 2; docs + CJK note + nbmake → Task 8. All spec sections covered.
- **Placeholder scan:** none; every code step shows concrete code.
- **Type consistency:** `_asset_labels(mode: str) -> list[str]`, `_labels_mode: str`, `_labels_mode_from_bool(bool) -> str`, `labels` property str, legacy shims bool — used consistently across Tasks 3/5/6/7.
- **Open verification during execution:** confirm exact exported frontier class names (Task 6 Step 1 note) and exact line numbers before editing (they may drift).
