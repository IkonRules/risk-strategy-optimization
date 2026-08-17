# Simulations.py
# High-level simulation engine that orchestrates full games using SimulationFunctions.
# Keeps mechanics (dice, battles, movement, reinforcements, rendering) in SimulationFunctions.

from __future__ import annotations

import logging
import random
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from project_risk.game_simulation import Board
from project_risk.game_simulation import Players
from project_risk.game_simulation import SimulationFunctions as SimFn


# --------------------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------------------

def setup_logging(level=logging.INFO):
    import logging
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    # Apply level to all related loggers
    for name in ("risk", "risk.sim"):
        logging.getLogger(name).setLevel(level)



# --------------------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------------------

class SimulationEngine:
    """
    Orchestrates a full Risk game using the mechanics provided by SimulationFunctions.
    This class is intentionally thin: it sequences turns and delegates *all* rules/logic
    to SimFn (reinforcements, battle choices, dice, movement, rendering).

    Usage:
        engine = SimulationEngine(players, rng_seed=123, max_turns=1_000)
        winner = engine.run()

    Optional rendering:
        engine = SimulationEngine(players, draw_initial=("map.png", "init.png"),
                                  draw_final=("map.png", "final.png"))
    """

    def __init__(
        self,
        players: Sequence[Players.Player],
        rng_seed: Optional[int] = None,
        max_turns: Optional[int] = None,
        draw_initial: Optional[Tuple[str, str]] = None,  # (base_map_path, output_path)
        draw_final: Optional[Tuple[str, str]] = None,    # (base_map_path, output_path)
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.players: List[Players.Player] = list(players)
        self.rng = random.Random(rng_seed) if rng_seed is not None else random
        self.max_turns = max_turns
        self.draw_initial = draw_initial
        self.draw_final = draw_final
        self.logger = logger or logging.getLogger("risk.sim")

    # ---- lifecycle -----------------------------------------------------------

    def setup(self) -> None:
        """Run any initial placement hooks and optional initial rendering."""
        # Filter to currently alive players (in case the board was preconfigured oddly)
        self.players = [p for p in self.players if SimFn.determine_if_player_in_game(p)]

        # Build a turn queue (SimFn provides the queue helper to keep consistency)
        self.queue = SimFn.initialize_playing_queue(self.players)

        # Initial placement phase (delegated to players if they implement it)
        SimFn.place_troops(self.queue, rng=self.rng)

        # Optional initial draw
        if self.draw_initial:
            base_map, out_path = self.draw_initial
            SimFn.save_game_state_to_image(self.players, base_map_image_path=base_map, output_image_path=out_path)

    def step_turn(self, player: Players.Player) -> Optional[Players.Player]:
        """
        Execute one full turn for `player`.
        Returns a Player if someone was eliminated during this turn; otherwise None.
        """
        # Reinforcements
        SimFn.add_troop_reinforcements(player, rng=self.rng)

        # Attack phase
        max_battles = getattr(player, "max_battles_per_turn", 5)
        eliminated_any: Optional[Players.Player] = None

        for _ in range(max_battles):
            pairs = list(SimFn.iter_battles(player))
            if not pairs:
                break

            # Allow the player to choose the battle; fallback is random
            chooser = getattr(player, "choose_battle", None)
            if callable(chooser):
                try:
                    atk, dfd = chooser(pairs, self.rng)
                    if (atk, dfd) not in pairs:
                        atk, dfd = pairs[self.rng.randrange(len(pairs))]
                except Exception:
                    self.logger.exception("Error in choose_battle; using random.")
                    atk, dfd = pairs[self.rng.randrange(len(pairs))]
            else:
                atk, dfd = pairs[self.rng.randrange(len(pairs))]

            eliminated = SimFn.battle(player, dfd._owner, atk, dfd, rng=self.rng)
            if eliminated is not None:
                eliminated_any = eliminated
                # Remove from internal state immediately
                if eliminated in self.players:
                    self.players.remove(eliminated)
                # Also purge from queue if present
                try:
                    while True:
                        self.queue.remove(eliminated)
                except ValueError:
                    pass

                if len(self.players) <= 1:
                    break  # game over

        # Fortify / reallocate step (player policy or default heuristic)
        SimFn.move_troops(player, rng=self.rng)

        return eliminated_any

    def run(self) -> Players.Player:
        """Run the simulation to completion (or until max_turns). Returns the winner."""
        self.setup()

        turn = 0
        while True:
            # --- Keep players' territory lists consistent with the board truth
            SimFn.reconcile_players(self.players)

            # Drop eliminated players from the roster
            self.players = [p for p in self.players if SimFn.determine_if_player_in_game(p)]

            # Also purge eliminated players from the queue
            try:
                # Rebuild queue with only alive players, preserving order of remaining
                alive_in_queue = [p for p in self.queue if p in self.players]
                self.queue.clear()
                for p in alive_in_queue:
                    self.queue.append(p)
            except Exception:
                pass

            # Victory or turn-cap check
            if len(self.players) <= 1:
                break
            if self.max_turns is not None and turn >= self.max_turns:
                break

            # Ensure queue has someone to move (can be empty after purges)
            if not self.queue:
                self.queue = SimFn.initialize_playing_queue(self.players)

            # --- Play one turn
            player = self.queue.popleft()
            if player not in self.players:
                # Player was eliminated during cleanup; skip turn
                continue

            self.step_turn(player)

            # If still alive, they go back to the end of the queue
            if SimFn.determine_if_player_in_game(player):
                self.queue.append(player)

            turn += 1

        # Final reconciliation & winner selection
        SimFn.reconcile_players(self.players)
        alive = [p for p in self.players if SimFn.determine_if_player_in_game(p)]

        if not alive:
            # Edge case: everyone eliminated due to a cap or pathological state.
            # Fall back to the player with most territories.
            winner = max(self.players, key=lambda p: len(getattr(p, "_territories", [])))
        else:
            winner = alive[0]

        self.logger.info("Winner: %s", winner._name)

        # Final render (IMPORTANT: render from self.players, not the queue)
        if self.draw_final:
            base_map, out_path = self.draw_final
            SimFn.save_game_state_to_image(self.players, base_map_image_path=base_map, output_image_path=out_path)

        return winner



# --------------------------------------------------------------------------------------
# Convenience top-level helpers (functional API)
# --------------------------------------------------------------------------------------

def run_simulation(
    players: Sequence[Players.Player],
    rng_seed: Optional[int] = None,
    max_turns: Optional[int] = None,
    draw_initial: Optional[Tuple[str, str]] = None,
    draw_final: Optional[Tuple[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Players.Player:
    """
    Run a single game and return the winner.
    Mirrors SimulationEngine but as a functional API.
    """
    engine = SimulationEngine(
        players=players,
        rng_seed=rng_seed,
        max_turns=max_turns,
        draw_initial=draw_initial,
        draw_final=draw_final,
        logger=logger,
    )
    return engine.run()


def run_many(
    players_factory: Callable[[], Sequence[Players.Player]],
    n_games: int,
    seeds: Optional[Sequence[int]] = None,
    **run_kwargs,
) -> List[Players.Player]:
    """
    Run many games for Monte Carlo analysis.
      - players_factory: callable producing a *fresh* set of players (and a fresh board state) each game
      - n_games: number of games to simulate
      - seeds: optional sequence of RNG seeds (length n_games)
      - run_kwargs: forwarded to run_simulation (e.g., max_turns=..., draw_initial=None, draw_final=None)
    Returns the list of winners (one per game).
    """
    winners: List[Players.Player] = []
    for i in range(n_games):
        players = list(players_factory())
        seed = None if seeds is None else seeds[i]
        winners.append(run_simulation(players, rng_seed=seed, **run_kwargs))
    return winners


# --------------------------------------------------------------------------------------
# Optional utilities useful when wiring a players_factory
# --------------------------------------------------------------------------------------

def reset_board_state() -> None:
    """
    Reset the global Board territory state (owners to None, troops to 0).
    Handy if your Board module holds singleton Territory objects.
    """
    for t in Board.all_territories_list:
        t._owner = None
        t._troops = 0


def random_initial_claim_and_seed(
    players: Sequence[Players.Player],
    troops_per_player: int = 20,
    rng: Optional[random.Random] = None,
) -> None:
    """
    Simple draft: randomly assign all territories to players in round-robin,
    then distribute troops_per_player as +1 increments randomly on owned territories.
    This is optional and only needed if you want to simulate from a blank map.
    """
    if rng is None:
        rng = random

    # Reset board
    reset_board_state()
    for p in players:
        p._territories.clear()
        p._continents.clear()

    terrs = list(Board.all_territories_list)
    rng.shuffle(terrs)

    # Round-robin claim
    for i, t in enumerate(terrs):
        owner = players[i % len(players)]
        t._owner = owner
        t._troops = 1
        owner._territories.append(t)

    # Seed extra troops
    for p in players:
        for _ in range(max(0, troops_per_player - len(p._territories))):
            t = p._territories[rng.randrange(len(p._territories))]
            t._troops += 1

