# Multi-Agent-Search
Third project of CentraleSupélec's Artificial Intelligence course

Design agents for the classic version of Pacman

https://centralesupelec.edunao.com/pluginfile.php/423959/course/section/60097/p3-multiagent.html?time=1715057118989

http://ai.berkeley.edu.

## Contents
- [Overview](#Overview)
- [Question1](#Question1)
- [Question2](#Question2)
- [Question3](#Question3)
- [Question4](#Question4)
- [Question5](#Question5)

the questions have been implemented in the file **File:** `multiAgents.py`

## Overview

This project implements a series of multi-agent search agents to play Pacman. The agents use different search and evaluation techniques to make real-time decisions while playing. The techniques include:

- Reflex Agent: Uses a simple evaluation function to choose actions.

- Minimax: Implements the minimax strategy to maximize the minimum expected score.

- Alpha-Beta Pruning: Optimizes the minimax algorithm with alpha-beta pruning to reduce the number of evaluated nodes.

- Expectimax: Models uncertainty in ghost actions with expected values.

- Evaluation Function: A function to evaluate game states considering multiple factors.


## Question1

Reflex Agent : This agent makes decisions based on a simple evaluation function considering the distance to food and the presence of ghosts.

**Class:** `ReflexAgent(Agent)`

To run:
```bash
python pacman.py -p ReflexAgent -l testClassic
```

## Question2

Implements the minimax algorithm to make decisions that maximize the minimum expected score.
 
**Class:** `MinimaxAgent(MultiAgentSearchAgent)`

To run:
```bash
python pacman.py -p MinimaxAgent -l trappedClassic -a depth=3
```

## Question3

Uses alpha-beta pruning to optimize the minimax algorithm, reducing the number of evaluated states.

**Class:** `AlphaBetaAgent(MultiAgentSearchAgent)`

To run:
```bash
python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
```

## Question4

Models uncertainty in ghost actions with expected values, choosing actions based on expected scores.

**Class:** `ExpectimaxAgent(MultiAgentSearchAgent)`

To run:
```bash
python pacman.py -p ExpectimaxAgent -l minimaxClassic -a depth=3
```

## Question5

Evaluation Function

**Class:** `betterEvaluationFunction`

This evaluation function considers:
    - Current game score

    - Distance to the closest food pellet

    - Distance to the closest ghost (penalizing being too close to active ghosts)

    - Number of remaining food pellets (penalizing more remaining food)

    - Distance to the closest capsule (encouraging eating capsules to turn ghosts scared)
    
    - Distance to the closest scared ghost (encouraging eating scared ghosts)

To run:
```bash
python autograder.py -q q5
```
