# COMP532 — Machine Learning with Bioinspired Optimisation: Coursework Assignment 2 (CA-2)

Repository Contents
COMP532-CA2/
├── README.md                          
├── COMP532_CA2_REPORT.docx            
├── LunarLander.ipynb      
└── chess_rl_agent.ipynb  

Problem 1 — Deep Reinforcement Learning: LunarLander-v3
Algorithm
Double Deep Q-Network (DDQN) with:

Experience replay buffer (100 000 transitions)
Soft target network updates (τ = 0.001)
ε-greedy exploration with exponential decay (1.0 → 0.01)
Huber loss + gradient clipping (max norm = 10)

Environment
LunarLander-v3 from OpenAI Gymnasium.

State space: 8 continuous features (position, velocity, angle, leg contacts)
Action space: 4 discrete actions (do nothing, left engine, main engine, right engine)
Solved criterion: Mean reward ≥ 200 over 100 consecutive episodes

Key Hyperparameters
ParameterValueReplay buffer size100 000Batch size64Learning rate5 × 10⁻⁴Discount factor γ0.99Soft update τ0.001Hidden layers[256, 256]ε start / end / decay1.0 / 0.01 / 0.995Update frequencyEvery 4 steps
How to Run
bash# Install dependencies
pip install gymnasium[box2d] torch matplotlib imageio

# Open notebook
jupyter notebook LunarLander.ipynb
The notebook will:

Train the DDQN agent for up to 1000 episodes (stops early when solved)
Plot reward and loss learning curves
Save and display an animated GIF of the trained agent playing

Outputs Generated
FileDescriptiontraining_curves.pngEpisode reward & Huber loss over trainingepsilon_decay.pngε exploration rate decay curvelunar_lander_agent.gifAnimated GIF of trained agent (3 episodes)ddqn_lunar_lander.pthSaved model weights

Problem 2 — Exploration vs. Exploitation
Addressed in full in COMP532_CA2_REPORT.docx (Section: Problem 2).
Summary: The exploration–exploitation dilemma arises because RL agents receive only evaluative feedback (how good an action was) rather than instructive feedback (what the correct action was). Pure exploitation risks locking onto sub-optimal policies; pure exploration never capitalises on gathered knowledge.
This assignment uses ε-greedy with exponential decay to balance both, with ε decaying from 1.0 (fully random) to 0.01 (mostly greedy) over ~900 episodes.

Optional Extension — LLM-Based Chess Agent
Method
An LLM-based chess agent powered by the Anthropic Claude API (claude-sonnet-4-20250514), evaluated against a random-move baseline over 40 games (20 as White, 20 as Black).
LLM Method Disclosures
AspectDetailModel / APIclaude-sonnet-4-20250514 via https://api.anthropic.com/v1/messagesPrompting designFEN + ASCII board + full legal-move list (UCI); single-token output requiredLegal move enforcementLegal moves listed explicitly in prompt; output validated against board.legal_movesInvalid move handlingUp to 3 retries with corrective feedback; fallback to piece-value heuristicReproducibilitytemperature=0 used; active Anthropic API key required; model version pinned
How to Run
bash# Install dependencies
pip install python-chess requests matplotlib imageio cairosvg Pillow

# Open notebook
jupyter notebook COMP532_CA2_Chess_Extension.ipynb

Note: The chess extension requires an active Anthropic API key. The notebook makes ~1 API call per LLM move. For a quick test, reduce N_EVAL_GAMES from 20 to 5.

Outputs Generated
FileDescriptionchess_results.pngWin/Draw/Loss bar charts (overall & by colour)chess_game_lengths.pngDistribution of game lengthschess_game.gifAnimated GIF of a full recorded game

Dependencies
PackageVersionPurposegymnasium[box2d]≥ 0.29LunarLander environmenttorch≥ 2.0Neural network & trainingmatplotlib≥ 3.7Plotting learning curvesimageio≥ 2.31GIF exportpython-chess≥ 1.10Chess board managementrequests≥ 2.31Anthropic API callscairosvg≥ 2.7SVG → PNG frame conversionPillow≥ 10.0Image processing for GIF
Install all at once:
bashpip install gymnasium[box2d] torch matplotlib imageio python-chess requests cairosvg Pillow

References

Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602
van Hasselt, Guez & Silver (2016). Deep RL with Double Q-learning. AAAI 2016
Sutton & Barto (2018). Reinforcement Learning: An Introduction. MIT Press
Farama Foundation (2024). Gymnasium Documentation. https://gymnasium.farama.org
Anthropic (2024). Claude API Documentation. https://docs.anthropic.com
