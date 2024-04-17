# Markov Chains

**Markov Property:** The future state depends only on the current state.

## Transition Diagram

- **Burger** to **Pizza**: 0.7
- **Burger** to **Burger**: 0.2
- **Burger** to **Hotdog**: 0.1
- **Pizza** to **Hotdog**: 0.7
- **Pizza** to **Pizza**: 0.2
- **Pizza** to **Burger**: 0.1

_Note:_ What they serve depends on what they served the previous day.

There is a 60% chance that today will be a pizza day if yesterday was a burger day.

## Transition Matrix (Not directly provided, but inferred)

|           | Burger | Pizza | Hotdog |
|-----------|--------|-------|--------|
| **Burger**| 0.2    | 0.7   | 0.1    |
| **Pizza** | 0.1    | 0.2   | 0.7    |
| **Hotdog**| (?)    | (?)   | (?)    |

(Note: Some values for the transition from Hotdog are missing)

## Probability Calculations

P(X_t = hotdog | X_3 = pizza) > 0.2

The outgoing state for the sum of the weights of all outgoing transitions must add up to 1 since they are probabilities.

- Pizza: 0.3 + 0.2 = 0.5
- Burger: 0.6 + 0.2 = 0.8
- Hotdog: 0.5 + 0.5 = 1.0

(Note: Some values may be off as they are inferred from incomplete data)
