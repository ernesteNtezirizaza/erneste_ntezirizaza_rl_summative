# Agent-Environment Diagram

This diagram summarizes how the reinforcement learning agent interacts with the posture correction mission environment.

```mermaid
flowchart LR
    A[Office worker posture state\nneck shoulder spine fatigue] --> B[Observation vector]
    B --> C[RL Agent\nDQN PPO or REINFORCE]
    C --> D{Action}
    D --> D1[Neck align cue]
    D --> D2[Shoulder reset cue]
    D --> D3[Lumbar support cue]
    D --> D4[Micro-break stretch]
    D --> D5[Walk break]
    D --> D6[Breathing reset]
    D --> D7[Ignore cue]
    D1 --> E[Environment dynamics]
    D2 --> E
    D3 --> E
    D4 --> E
    D5 --> E
    D6 --> E
    D7 --> E
    E --> F[Updated posture quality + fatigue]
    F --> G[Reward]
    G --> C

    subgraph Reward Components
        R1[Posture quality improvement]
        R2[Fatigue control]
        R3[Penalty for ignoring high risk]
        R4[Bonus for timely breaks]
    end

    F --> R1
    F --> R2
    D7 --> R3
    D4 --> R4
```

## Description

- State: a continuous representation of neck alignment quality, shoulder symmetry quality, spinal support quality, fatigue, time at desk, and recent trend.
- Action: the agent chooses one ergonomic intervention at each step.
- Transition: posture naturally drifts due to fatigue and static sitting; actions counter this drift with varying strengths.
- Reward: shaped to encourage stable posture improvement, prevent fatigue escalation, and use breaks at the right time.
- Goal: maximize cumulative ergonomic score while completing a work session without severe posture degradation.
