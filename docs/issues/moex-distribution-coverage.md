# Missing distributions in MOEX ETF data

The 2026-10-26 API probe returned header-only dividend CSV responses for
`TPAY.MOEX`, `FLOW.MOEX`, and `AKIE.MOEX`. All three have valid ETF symbol
metadata. The `SBER.MOEX` control returned 26 payments, through 2026-07-20.

When `Asset` initializes one of these three funds, it checks its full API
distribution history and emits `okama.MissingDistributionsWarning` if that
history is empty. AssetList, Portfolio and EfficientFrontier construction also
surface the warning when they construct these Asset objects. Other symbols
receive no extra request. The input returns and dividend calculations are not
rewritten and no payments or dates are inferred.

The warning is deliberately limited to the reproduced cases. Its absence is not
a guarantee of complete distribution coverage or corrected total returns. It
stops when the API supplies payment data, which alone does not certify a complete
history or an adjusted return series. A failed dividend request propagates the
normal API error rather than classifying the failure as empty history.

[okama-API #545](https://github.com/mbk-dev/okama-API/issues/545) tracks the data
collection, provenance and machine-readable coverage contract. It coordinates
with the existing date-semantics and adjusted-close tasks. The library warning
is a mitigation; [okama #125](https://github.com/mbk-dev/okama/issues/125) remains
open until the underlying distributions and total-return data are corrected.

To reproduce the data probe without library caching:

```bash
for symbol in TPAY.MOEX FLOW.MOEX AKIE.MOEX SBER.MOEX; do
  curl --fail --silent --show-error "https://api.okama.io/api/symbol/$symbol"
  curl --fail --silent --show-error "https://api.okama.io/api/ts/dividends/$symbol"
done
```
