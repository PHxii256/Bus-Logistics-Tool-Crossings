## Benchmark Comparison Results

### Route Count: 4/10 Exact Match

| Instance     | Students | Lewis Routes | Ours | Match   |
| ------------ | -------- | ------------ | ---- | ------- |
| Porthcawl    | 66       | 1            | 1    | **YES** |
| Suffolk      | 209      | 3            | 3    | **YES** |
| MiltonKeynes | 274      | 4            | 4    | **YES** |
| Bridgend     | 381      | 6            | 6    | **YES** |
| Cardiff      | 156      | 2            | 3    | +1      |
| Edinburgh-2  | 320      | 4            | 5    | +1      |
| Canberra     | 499      | 7            | 8    | +1      |
| Edinburgh-1  | 680      | 9            | 10   | +1      |
| Adelaide     | 565      | 8            | 9    | +1      |
| Brisbane     | 757      | 10           | 11   | +1      |

### Journey Time: +189% to +365% Gap

The gaps are massive and **expected**. Here's why:

**Your system visits ~20 stops per route** (one virtual stop per student address). Lewis selects **5–8 stops per route** from a predefined candidate set, clustering multiple students at each stop. Fewer stops = dramatically shorter driving time.

This isn't a bug — it's a fundamentally different problem formulation:

|                      | Lewis (Stop Selection SBRP)          | Your System                          |
| -------------------- | ------------------------------------ | ------------------------------------ |
| **Stops**            | Pick subset from candidates          | Virtual stop per student             |
| **Students walk to** | Nearest selected stop (up to 1.6 km) | Their own doorstep stop              |
| **Stops per route**  | ~5–8                                 | ~15–33                               |
| **Key tradeoff**     | Students walk more, buses drive less | Students walk less, buses drive more |

### What This Means for Your Paper

The benchmarks demonstrate:

1. **100% student coverage** on all 10 instances (feasibility validated)
2. **Route counts within +0 to +1** of Lewis (fleet sizing works)
3. **Scalability** from 66 to 757 students (0.9s to 500s runtime)
4. Your system **cannot be compared on journey time** because it solves a different variant — it prioritises minimising student walking (door-to-door pickup) over minimising bus driving time
