# Matched four-arm results

Each cell is full-system OA / BA (%), not an isolated module accuracy.
M1 and M4 remain active in every arm. Each row uses its own matched reference.

| Dataset | Reference | M2 only | M3 only | M2+M3 | Joint delta OA / BA (pp) |
| --- | --- | --- | --- | --- | --- |
| self24ports | 56.653819 / 50.382098 | 56.679312 / 50.258322 | 56.756475 / 50.577416 | 56.961200 / 50.777800 | +0.307381 / +0.395702 |
| F0921 | 29.634353 / 25.106684 | 29.829289 / 25.003473 | 29.326474 / 25.434966 | 29.758639 / 25.431755 | +0.124286 / +0.325071 |
| OSS1 | 43.066148 / 32.330577 | 43.284047 / 31.970674 | 43.003891 / 31.848632 | 43.299611 / 31.758511 | +0.233463 / -0.572066 |
| OSS2 | 32.382505 / 29.938364 | 32.442271 / 29.971866 | 32.882369 / 30.318171 | 33.137734 / 30.556508 | +0.755230 / +0.618145 |
| pacs | 77.809164 / 79.635512 | 77.813689 / 79.636341 | 77.349439 / 79.266995 | 77.280367 / 79.184320 | -0.528798 / -0.451192 |
| officehome | 67.128250 / 65.392589 | 67.096863 / 65.372944 | 66.252547 / 64.557644 | 65.973213 / 64.265466 | -1.155038 / -1.127123 |

Complete dataset groups: 6/6.

## Expert versus full-system changes

Expert scores evaluate the expert alone on every query, without the 5% handoff rule.

| Dataset | Reference expert OA / BA | Joint expert OA / BA | Expert delta OA / BA (pp) |
| --- | --- | --- | --- |
| self24ports | 58.425285 / 52.894549 | 60.999502 / 54.480111 | +2.574218 / +1.585562 |
| F0921 | 31.055945 / 25.982723 | 31.874580 / 27.147474 | +0.818635 / +1.164750 |
| OSS1 | 50.817121 / 30.551436 | 52.420233 / 28.865268 | +1.603113 / -1.686168 |
| OSS2 | 34.947025 / 30.909838 | 39.782668 / 32.608236 | +4.835642 / +1.698398 |
| pacs | 79.058251 / 80.890574 | 72.908372 / 74.815983 | -6.149879 / -6.074592 |
| officehome | 64.835052 / 63.428996 | 54.897486 / 53.497736 | -9.937565 / -9.931260 |

Ports: 23 eligible targets, equal-port averaging; Melbourne has 64 rows and no remaining query.
Ports use the same float16-prefix initial tail in all four arms; old-cache M2 is reported separately.
PACS/OfficeHome: four-domain averages of rebuilt source RGB encoders, not historical feature replay.
F0921: reuse the completed matched pilot. OSS1/OSS2: original reference replay is exact.
Shared64 target labels, five support draws, at most 5% handoffs; one neural-training seed.
No target-query model selection; historical development data, not blind confirmation.
FUSAR, FMoW and Camelyon17 are not completed current-protocol joint experiments in this table.
